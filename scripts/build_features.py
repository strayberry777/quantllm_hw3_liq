#!/usr/bin/env python3
"""
1 秒网格采样：50 只股票的五档盘口（无前视）+ 逐笔订单流累计量。

网格：09:30:00 ~ 11:30:00、13:00:00 ~ 14:57:00，每秒一个点。
输出：outputs/features/grid_1s.parquet

用法：python scripts/build_features.py [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lob.events import build_events, load_tables  # noqa: E402
from lob.flow import sample_grid  # noqa: E402

_DATA: dict = {}


def make_grid() -> np.ndarray:
    def seq(h0, m0, h1, m1):
        start = (h0 * 3600 + m0 * 60)
        end = (h1 * 3600 + m1 * 60)
        s = np.arange(start, end + 1)
        return (s // 3600) * 10_000_000 + (s // 60 % 60) * 100_000 + (s % 60) * 1000
    return np.concatenate([seq(9, 30, 11, 30), seq(13, 0, 14, 57)]).astype(np.int64)


def init_worker(data_dir: str, date: str) -> None:
    o, e, _ = load_tables(Path(data_dir), date)
    _DATA["O"] = {k[0]: v for k, v in o.partition_by("Symbol", as_dict=True).items()}
    _DATA["E"] = {k[0]: v for k, v in e.partition_by("Symbol", as_dict=True).items()}


def run_symbol(sym: int) -> pl.DataFrame:
    ev = build_events(_DATA["O"][sym], _DATA["E"][sym], sym)
    grid = make_grid()
    out = sample_grid(ev, grid)
    return pl.DataFrame({"Symbol": np.full(len(grid), sym, dtype=np.int64), "Time": grid, **out})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "raw/sse50_20260923"))
    ap.add_argument("--date", default="20260923")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/features/grid_1s.parquet")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    symbols = sorted(pl.read_parquet(Path(args.data_dir) / f"snp_{args.date}.parquet", columns=["Symbol"])["Symbol"].unique().to_list())
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(args.data_dir, args.date)) as pool:
        frames = []
        for i, df in enumerate(pool.imap_unordered(run_symbol, symbols), 1):
            frames.append(df)
            print(f"[{i:02d}/{len(symbols)}] {df['Symbol'][0]} ({time.time() - t0:.0f}s)", flush=True)
    pl.concat(frames).sort(["Symbol", "Time"]).write_parquet(args.out, compression="zstd")
    print(f"写入 {args.out}，{sum(f.height for f in frames):,} 行，{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
