#!/usr/bin/env python3
"""
复现检查：用参考仓库自己的校验口径（含 14:57~15:00 收盘集合竞价、两端闭区间），
以 time_le 策略重算 50 只股票的 L1 价+量吻合率，与参考仓库公布的
reconstruction_validation_20260923.csv 逐只比对。

用法：python scripts/reproduce_reference.py --reference <参考仓库的 reconstruction_validation_20260923.csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lob.events import build_events, load_tables  # noqa: E402
from lob.replay import cut_points, replay  # noqa: E402
from lob.validate import compare, snapshot_matrix  # noqa: E402


def ref_window(t: np.ndarray) -> np.ndarray:
    return ((t >= 93_000_000) & (t <= 113_000_000)) | ((t >= 130_000_000) & (t <= 150_000_000))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "raw/sse50_20260923")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/reconstruction/reference_reproduction.csv")
    args = ap.parse_args()

    o, e, s = load_tables(args.data_dir, "20260923")
    ref = pd.read_csv(args.reference, encoding="utf-8-sig")
    rows = []
    for sym in sorted(s["Symbol"].unique().to_list()):
        ev = build_events(o.filter(pl.col("Symbol") == sym), e.filter(pl.col("Symbol") == sym), sym)
        ss = s.filter(pl.col("Symbol") == sym).sort("Time")
        st = ss["Time"].to_numpy().astype(np.int64)
        zeros = np.zeros(len(st), dtype=np.int64)
        cuts, _ = cut_points(ev, st, zeros, zeros, zeros, "time_le")
        out, _ = replay(ev, cuts)
        c = compare(out, snapshot_matrix(ss))
        m = ref_window(st)
        rows.append({"Symbol": sym, "Checked": int(m.sum()),
                     "L1PriceMatch_ours": float(c["l1_price"][m].mean()),
                     "L1FullMatch_ours": float(c["l1_full"][m].mean())})
    ours = pd.DataFrame(rows)
    merged = ours.merge(ref[["Symbol", "CheckedSnapshots", "L1PriceMatchRate", "L1FullMatchRate"]], on="Symbol")
    merged["AbsDiffL1Full"] = (merged["L1FullMatch_ours"] - merged["L1FullMatchRate"]).abs()
    merged["AbsDiffL1Price"] = (merged["L1PriceMatch_ours"] - merged["L1PriceMatchRate"]).abs()
    merged.to_csv(args.out, index=False)
    print(merged[["Symbol", "Checked", "CheckedSnapshots", "L1FullMatch_ours", "L1FullMatchRate"]].head(8).to_string(index=False))
    print(f"\n股票数 {len(merged)}；检查快照数一致 {int((merged.Checked == merged.CheckedSnapshots).sum())}/{len(merged)}")
    print(f"L1 价+量：我们 {merged.L1FullMatch_ours.mean():.4f}  参考 {merged.L1FullMatchRate.mean():.4f}  "
          f"最大逐只差 {merged.AbsDiffL1Full.max():.6f}")
    print(f"L1 价格：  我们 {merged.L1PriceMatch_ours.mean():.4f}  参考 {merged.L1PriceMatchRate.mean():.4f}  "
          f"最大逐只差 {merged.AbsDiffL1Price.max():.6f}")


if __name__ == "__main__":
    main()
