#!/usr/bin/env python3
"""
订单流因子研究（按 research/PROTOCOL.md 执行）。

    python scripts/factor_research.py --stage discovery   # 只用上午：定方向、选窗口、冻结入选名单
    python scripts/factor_research.py --stage holdout     # 读冻结名单，下午只运行一次（台账记次）
    python scripts/factor_research.py --stage lookahead   # 快照前视偏差检验（方法学，不参与选择）

输出：outputs/research/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/research"
GRID = ROOT / "outputs/features/grid_1s.parquet"
HORIZONS = (10, 30, 60)
PRIMARY = 30
WINDOWS = (10, 30)
BOOK = ("imb1", "imb5", "micro")
FLOW = ("ofi", "tfi", "big", "cxl", "add")
DISCOVERY = (93_100_000, 112_900_000)
HOLDOUT = (130_100_000, 145_500_000)
T_GATE, POS_GATE = 3.0, 0.70
SEED = 20260923


# ---------------------------------------------------------------- 特征
def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.where(b != 0)).fillna(0.0)


def build_panel(path: Path = GRID) -> pd.DataFrame:
    d = pd.read_parquet(path).sort_values(["Symbol", "Time"]).reset_index(drop=True)
    d["mid"] = (d.BidPrice1 + d.AskPrice1) / 2
    g = d.groupby("Symbol", sort=False)

    d["imb1"] = ratio(d.BidVolume1 - d.AskVolume1, d.BidVolume1 + d.AskVolume1)
    bw = sum(d[f"BidVolume{i}"] / i for i in range(1, 6))
    aw = sum(d[f"AskVolume{i}"] / i for i in range(1, 6))
    d["imb5"] = ratio(bw - aw, bw + aw)
    micro_px = ratio(d.AskPrice1 * d.BidVolume1 + d.BidPrice1 * d.AskVolume1, d.BidVolume1 + d.AskVolume1)
    d["micro"] = ratio(micro_px - d.mid, d.AskPrice1 - d.BidPrice1)

    depth = d.BidVolume1 + d.AskVolume1
    for w in WINDOWS:
        dd = {c: g[c].diff(w) for c in ("ofi", "aggr_buy_v", "aggr_sell_v", "big_buy_v", "big_sell_v",
                                        "cxl_bid_near", "cxl_ask_near", "add_bid_near", "add_ask_near")}
        tot = dd["aggr_buy_v"] + dd["aggr_sell_v"]
        d[f"ofi_{w}"] = ratio(dd["ofi"], depth)
        d[f"tfi_{w}"] = ratio(dd["aggr_buy_v"] - dd["aggr_sell_v"], tot)
        d[f"big_{w}"] = ratio(dd["big_buy_v"] - dd["big_sell_v"], tot)
        d[f"cxl_{w}"] = ratio(dd["cxl_ask_near"] - dd["cxl_bid_near"], dd["cxl_ask_near"] + dd["cxl_bid_near"])
        d[f"add_{w}"] = ratio(dd["add_bid_near"] - dd["add_ask_near"], dd["add_bid_near"] + dd["add_ask_near"])

    for h in HORIZONS:
        d[f"ret_{h}"] = g["mid"].shift(-h) / d["mid"] - 1
    d["block"] = (d.Time // 10_000_000) * 100 + (d.Time // 100_000 % 100) // 30 * 30
    return d


def segment(d: pd.DataFrame, seg: tuple[int, int]) -> pd.DataFrame:
    return d[(d.Time >= seg[0]) & (d.Time <= seg[1])]


# ---------------------------------------------------------------- 指标
def stock_ic(d: pd.DataFrame, factor: str, label: str) -> pd.Series:
    def one(z: pd.DataFrame) -> float:
        z = z[[factor, label]].dropna()
        if z[factor].nunique() < 2 or z[label].nunique() < 2:
            return np.nan
        return z[factor].rank().corr(z[label].rank())
    return d.groupby("Symbol").apply(one, include_groups=False).dropna()


def block_bootstrap(d: pd.DataFrame, factor: str, label: str, n: int = 2000) -> tuple[float, float]:
    """30 分钟时间块 bootstrap：每块内逐股票 IC 的均值 → 块级序列重抽样。"""
    blocks = d.groupby("block").apply(lambda z: stock_ic(z, factor, label).mean(), include_groups=False).dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    draw = blocks[rng.integers(0, len(blocks), (n, len(blocks)))].mean(axis=1)
    return float(np.quantile(draw, 0.025)), float(np.quantile(draw, 0.975))


def summarize(ic: pd.Series) -> dict:
    return {"MeanIC": float(ic.mean()), "TStat": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))),
            "PositiveShare": float((ic > 0).mean()), "Stocks": int(len(ic))}


def zscore(d: pd.DataFrame, cols: list[str], ref: pd.DataFrame) -> pd.DataFrame:
    """逐股票标准化，均值与标准差只取自发现段 ref。"""
    mu = ref.groupby("Symbol")[cols].mean()
    sd = ref.groupby("Symbol")[cols].std().replace(0, np.nan)
    z = (d[cols] - mu.reindex(d.Symbol).to_numpy()) / sd.reindex(d.Symbol).to_numpy()
    return z.fillna(0.0)


# ---------------------------------------------------------------- 阶段
def discovery(d: pd.DataFrame) -> None:
    disc = segment(d, DISCOVERY)
    rows = []
    candidates = list(BOOK) + [f"{f}_{w}" for f in FLOW for w in WINDOWS]
    for h in HORIZONS:
        for c in candidates:
            s = summarize(stock_ic(disc, c, f"ret_{h}"))
            rows.append({"Factor": c, "Horizon": h, **s})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "discovery_ic.csv", index=False)

    prim = res[res.Horizon == PRIMARY].set_index("Factor")
    selected, decisions = [], []
    for f in FLOW:
        passing = []
        for w in WINDOWS:
            r = prim.loc[f"{f}_{w}"]
            ok = r.MeanIC > 0 and r.TStat >= T_GATE and r.PositiveShare >= POS_GATE
            decisions.append({"Factor": f"{f}_{w}", "MeanIC": r.MeanIC, "TStat": r.TStat, "PositiveShare": r.PositiveShare, "Pass": bool(ok)})
            if ok:
                passing.append((w, r.MeanIC))
        if len(passing) == 2:
            (w1, ic1), (w2, ic2) = passing
            pick = 10 if abs(ic1 - ic2) < 0.005 else (w1 if ic1 > ic2 else w2)
            selected.append(f"{f}_{pick}")
        elif len(passing) == 1:
            selected.append(f"{f}_{passing[0][0]}")
    pd.DataFrame(decisions).to_csv(OUT / "discovery_decisions.csv", index=False)

    frozen = {"frozen_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "primary_horizon": PRIMARY,
              "book": ["imb5", "micro"], "flow": selected, "t_gate": T_GATE, "pos_gate": POS_GATE}
    (OUT / "selection_frozen.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(res[res.Horizon == PRIMARY].round(4).to_string(index=False))
    print("\n入选订单流因子：", selected)


def holdout(d: pd.DataFrame) -> None:
    frozen_path = OUT / "selection_frozen.json"
    if not frozen_path.exists():
        raise SystemExit("先运行 --stage discovery 冻结入选名单")
    sel = json.loads(frozen_path.read_text(encoding="utf-8"))
    ledger = OUT / "holdout_ledger.json"
    runs = json.loads(ledger.read_text()) if ledger.exists() else []
    runs.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ledger.write_text(json.dumps(runs, indent=2))

    disc, hold = segment(d, DISCOVERY), segment(d, HOLDOUT).copy()
    book, flow = sel["book"], sel["flow"]
    z = zscore(hold, book + flow, disc)
    hold["BOOK"] = z[book].mean(axis=1)
    hold["FLOW"] = z[flow].mean(axis=1) if flow else 0.0
    hold["ALL"] = z[book + flow].mean(axis=1)

    rows, paired = [], []
    for h in HORIZONS:
        lab = f"ret_{h}"
        ics = {}
        for c in book + flow + ["BOOK", "FLOW", "ALL"]:
            ic = stock_ic(hold, c, lab)
            ics[c] = ic
            lo, hi = block_bootstrap(hold, c, lab) if c in ("BOOK", "FLOW", "ALL") and h == PRIMARY else (np.nan, np.nan)
            rows.append({"Factor": c, "Horizon": h, **summarize(ic), "BlockCI95Low": lo, "BlockCI95High": hi})
        diff = (ics["ALL"] - ics["BOOK"]).dropna()
        paired.append({"Horizon": h, "MeanDiff_ALL_minus_BOOK": float(diff.mean()),
                       "PairedT": float(diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))),
                       "ShareStocksImproved": float((diff > 0).mean())})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "holdout_ic.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "holdout_increment.csv", index=False)

    # 逐股票 IC（主周期）供报告与图
    per = pd.DataFrame({c: stock_ic(hold, c, f"ret_{PRIMARY}") for c in ["BOOK", "FLOW", "ALL"]})
    per.to_csv(OUT / "holdout_stock_ic_30s.csv")
    print(res.round(4).to_string(index=False))
    print(pd.DataFrame(paired).round(4).to_string(index=False))
    print(f"留出段运行次数：{len(runs)}")


def lookahead(d: pd.DataFrame) -> None:
    """同一组盘口形状因子：交易所快照（标签 T） vs 无前视盘口（Time < T）。"""
    import polars as pl
    snp = pl.read_parquet(ROOT / "raw/sse50_20260923/snp_20260923.parquet").to_pandas()
    snp = snp[((snp.Time >= 93_100_000) & (snp.Time <= 112_900_000)) | ((snp.Time >= 130_100_000) & (snp.Time <= 145_500_000))]
    snp = snp[(snp.BidPrice1 > 0) & (snp.AskPrice1 > snp.BidPrice1)].copy()
    for c in [c for c in snp.columns if "Price" in c and c[-1].isdigit()]:
        snp[c] = (snp[c] * 100).round()

    def shape(x: pd.DataFrame) -> pd.DataFrame:
        o = pd.DataFrame(index=x.index)
        o["imb1"] = ratio(x.BidVolume1 - x.AskVolume1, x.BidVolume1 + x.AskVolume1)
        bw = sum(x[f"BidVolume{i}"] / i for i in range(1, 6))
        aw = sum(x[f"AskVolume{i}"] / i for i in range(1, 6))
        o["imb5"] = ratio(bw - aw, bw + aw)
        mid = (x.BidPrice1 + x.AskPrice1) / 2
        mp = ratio(x.AskPrice1 * x.BidVolume1 + x.BidPrice1 * x.AskVolume1, x.BidVolume1 + x.AskVolume1)
        o["micro"] = ratio(mp - mid, x.AskPrice1 - x.BidPrice1)
        return o

    raw = shape(snp).add_prefix("snap_")
    raw[["Symbol", "Time"]] = snp[["Symbol", "Time"]]
    ours = d[["Symbol", "Time", "imb1", "imb5", "micro"] + [f"ret_{h}" for h in HORIZONS]]
    m = raw.merge(ours, on=["Symbol", "Time"], how="inner")
    rows = []
    for h in HORIZONS:
        for f in ("imb1", "imb5", "micro"):
            a = summarize(stock_ic(m, f"snap_{f}", f"ret_{h}"))
            b = summarize(stock_ic(m, f, f"ret_{h}"))
            rows.append({"Factor": f, "Horizon": h, "IC_snapshot_labelT": a["MeanIC"], "IC_noLookahead": b["MeanIC"],
                         "Inflation": a["MeanIC"] - b["MeanIC"], "InflationRatio": a["MeanIC"] / b["MeanIC"] if b["MeanIC"] else np.nan,
                         "Rows": len(m)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "lookahead_bias.csv", index=False)
    print(res.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("discovery", "holdout", "lookahead"), required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = build_panel()
    {"discovery": discovery, "holdout": holdout, "lookahead": lookahead}[args.stage](d)


if __name__ == "__main__":
    main()
