#!/usr/bin/env python3
"""
50 只股票 × 6 种对齐策略的盘口重建与逐档校验。

输出（outputs/reconstruction/）：
  validation_by_symbol.csv     每只股票 × 每种策略的吻合率
  validation_summary.csv       各策略全样本汇总（逐快照池化）
  offsets.csv                  每只股票的相位偏移：全天估计 / 仅上午估计，及一致率
  residuals.csv                推荐策略下仍不吻合的快照及归因
  reconstructed_orderbook_20260923.parquet   推荐策略的十档盘口（每个快照时间点一行）

用法：python scripts/reconstruct.py [--data-dir raw/sse50_20260923] [--workers 2]
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
from lob.replay import POLICIES, cut_points, estimate_offset, replay, totals_path, trade_gap, totals_candidates  # noqa: E402
from lob.validate import LEVELS, compare, in_continuous, snapshot_matrix  # noqa: E402

RECOMMENDED = "phase_totals"
MORNING = ((93_000_000, 113_000_000),)
AFTERNOON = ((130_000_000, 145_700_000),)
_DATA: dict = {}


def init_worker(data_dir: str, date: str) -> None:
    o, e, s = load_tables(Path(data_dir), date)
    _DATA["O"] = {k[0]: v for k, v in o.partition_by("Symbol", as_dict=True).items()}
    _DATA["E"] = {k[0]: v for k, v in e.partition_by("Symbol", as_dict=True).items()}
    _DATA["S"] = {k[0]: v.sort("Time") for k, v in s.partition_by("Symbol", as_dict=True).items()}


def run_symbol(sym: int) -> dict:
    t0 = time.time()
    ev = build_events(_DATA["O"][sym], _DATA["E"][sym], sym)
    ss = _DATA["S"][sym]
    st = ss["Time"].to_numpy().astype(np.int64)
    mi = ss["MatchItem"].to_numpy().astype(np.int64)
    tb = ss["TotalBidVolume"].to_numpy()
    ta = ss["TotalAskVolume"].to_numpy()
    src = snapshot_matrix(ss)
    cont = in_continuous(st)
    am, pm = in_continuous(st, MORNING), in_continuous(st, AFTERNOON)
    path = totals_path(ev)

    off, share = estimate_offset(ev, st, mi, tb, ta, path, cont)
    off_am, share_am = estimate_offset(ev, st, mi, tb, ta, path, am)

    rows, pooled = [], {}
    keep = None
    for policy in POLICIES:
        cuts, how = cut_points(ev, st, mi, tb, ta, policy, path, off)
        out, book = replay(ev, cuts)
        c = compare(out, src)
        rows.append({
            "Symbol": sym, "Policy": policy, "Events": len(ev), "Checked": int(cont.sum()),
            "L1PriceMatch": float(c["l1_price"][cont].mean()),
            "L1FullMatch": float(c["l1_full"][cont].mean()),
            "AllLevelPriceMatch": float(c["levels_price"][cont].mean() / LEVELS),
            "AllLevelFullMatch": float(c["levels_full"][cont].mean() / LEVELS),
            "Top10AllExact": float(c["all10_full"][cont].mean()),
            "TotalsHit": float(how[cont].mean()),
        })
        pooled[policy] = {k: v[cont] for k, v in c.items()}
        if policy == RECOMMENDED:
            keep = (cuts, how, out, c, book)

    # 只用上午估计的偏移去重建下午：检验偏移是否全天稳定（时间外推）
    cuts_pm, _ = cut_points(ev, st, mi, tb, ta, "phase", path, off_am)
    out_pm, _ = replay(ev, cuts_pm)
    c_pm = compare(out_pm, src)
    cuts_pm2, _ = cut_points(ev, st, mi, tb, ta, "phase_totals", path, off_am)
    out_pm2, _ = replay(ev, cuts_pm2)
    c_pm2 = compare(out_pm2, src)

    cuts, how, out, c, book = keep
    # 残差归因
    lo, hi = trade_gap(ev, mi)
    resid = []
    for j in np.flatnonzero(cont & ~c["all10_full"]):
        ncand = len(totals_candidates(path, lo, hi, tb, ta, j))
        r = out[j].reshape(LEVELS, 4)
        s = src[j].reshape(LEVELS, 4)
        first_bad = int(np.flatnonzero(~((r == s).all(axis=1)))[0]) + 1
        if ncand == 0:
            cause = "no_totals_match"
        elif ncand > 1:
            cause = "ambiguous_cut"
        else:
            cause = "unique_cut_but_levels_differ"
        resid.append({"Symbol": sym, "Time": int(st[j]), "Cause": cause, "Candidates": ncand,
                      "FirstBadLevel": first_bad, "L1Full": bool(c["l1_full"][j])})

    book_df = pl.DataFrame({
        "Symbol": [f"{sym:06d}"] * len(st), "Time": st, "Cut": cuts, "TotalsHit": how,
        **{f"{side}{kind}{lv}": (out[:, (lv - 1) * 4 + off_i] / (100 if kind == "Price" else 1))
           for lv in range(1, LEVELS + 1)
           for off_i, (side, kind) in enumerate([("Bid", "Price"), ("Bid", "Volume"), ("Ask", "Price"), ("Ask", "Volume")])},
        "MatchedFullLevels": c["levels_full"], "L1FullMatch": c["l1_full"], "Top10AllExact": c["all10_full"],
        "InContinuous": cont,
    })
    return {
        "rows": rows, "pooled": pooled, "resid": resid, "book": book_df,
        "offset": {"Symbol": sym, "OffsetMs": off, "Consistency": share,
                   "OffsetMsMorningOnly": off_am, "ConsistencyMorningOnly": share_am,
                   "PM_phase_Top10AllExact_withMorningOffset": float(c_pm["all10_full"][pm].mean()),
                   "PM_phaseTotals_Top10AllExact_withMorningOffset": float(c_pm2["all10_full"][pm].mean()),
                   "Anomalies": sum(book.anomalies.values())},
        "seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "raw/sse50_20260923"))
    ap.add_argument("--date", default="20260923")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/reconstruction")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    symbols = sorted(pl.read_parquet(Path(args.data_dir) / f"snp_{args.date}.parquet", columns=["Symbol"])["Symbol"].unique().to_list())
    t0 = time.time()
    # spawn 而非 fork：polars 的线程池在 fork 出的子进程里会死锁
    with get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(args.data_dir, args.date)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_symbol, symbols), 1):
            results.append(r)
            print(f"[{i:02d}/{len(symbols)}] {r['offset']['Symbol']} offset={r['offset']['OffsetMs']}ms "
                  f"top10={r['rows'][-1]['Top10AllExact']:.4f} ({r['seconds']:.1f}s)", flush=True)
    results.sort(key=lambda r: r["offset"]["Symbol"])

    by_sym = pl.DataFrame([row for r in results for row in r["rows"]])
    by_sym.write_csv(args.out / "validation_by_symbol.csv")

    summary = []
    for p in POLICIES:
        cat = {k: np.concatenate([r["pooled"][p][k] for r in results]) for k in results[0]["pooled"][p]}
        sub = by_sym.filter(pl.col("Policy") == p)
        summary.append({
            "Policy": p,
            "Snapshots": len(cat["l1_full"]),
            "L1PriceMatch": float(cat["l1_price"].mean()),
            "L1FullMatch": float(cat["l1_full"].mean()),
            "AllLevelFullMatch": float(cat["levels_full"].mean() / LEVELS),
            "Top10AllExact": float(cat["all10_full"].mean()),
            "WorstSymbolTop10": float(sub["Top10AllExact"].min()),
            "MedianSymbolTop10": float(sub["Top10AllExact"].median()),
            "SymbolMeanL1Full": float(sub["L1FullMatch"].mean()),
        })
    pl.DataFrame(summary).write_csv(args.out / "validation_summary.csv")
    pl.DataFrame([r["offset"] for r in results]).write_csv(args.out / "offsets.csv")
    resid = [x for r in results for x in r["resid"]]
    pl.DataFrame(resid, schema={"Symbol": pl.Int64, "Time": pl.Int64, "Cause": pl.Utf8, "Candidates": pl.Int64,
                                "FirstBadLevel": pl.Int64, "L1Full": pl.Boolean}).write_csv(args.out / "residuals.csv")
    pl.concat([r["book"] for r in results]).write_parquet(
        args.out / "reconstructed_orderbook_20260923.parquet", compression="zstd")

    print(pl.DataFrame(summary))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
