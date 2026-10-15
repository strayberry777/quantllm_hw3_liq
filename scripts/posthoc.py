#!/usr/bin/env python3
"""
事后解释性分析（在留出段运行之后写的，不参与任何选择，结论只作解释）。

1. 快照时间错位的分解：按「T 到 T+偏移之间中间价是否变化」把样本分两组，
   比较快照特征与无前视特征的 IC
2. 订单流为什么没有增量：FLOW 与 BOOK 的相关，以及 FLOW 对 BOOK 残差化后的偏 IC（留出段）
3. micro ≡ imb1/2 的数值核对

输出：outputs/research/posthoc_*.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from factor_research import DISCOVERY, HOLDOUT, OUT, build_panel, ratio, segment, stock_ic, summarize, zscore  # noqa: E402


def timing_decomposition(d: pd.DataFrame) -> pd.DataFrame:
    snp = pl.read_parquet(ROOT / "raw/sse50_20260923/snp_20260923.parquet",
                          columns=["Symbol", "Time", "BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1"]).to_pandas()
    snp = snp[(snp.BidPrice1 > 0) & (snp.AskPrice1 > snp.BidPrice1)].copy()
    snp["snap_mid"] = (snp.BidPrice1 + snp.AskPrice1) / 2 * 100
    snp["snap_imb1"] = ratio(snp.BidVolume1 - snp.AskVolume1, snp.BidVolume1 + snp.AskVolume1)
    m = snp.merge(d[["Symbol", "Time", "mid", "imb1", "ret_30"]], on=["Symbol", "Time"])
    m = m[((m.Time >= DISCOVERY[0]) & (m.Time <= DISCOVERY[1])) | ((m.Time >= HOLDOUT[0]) & (m.Time <= HOLDOUT[1]))]
    m["mid_moved"] = (m.snap_mid.round() != m.mid.round())
    rows = []
    for moved, g in m.groupby("mid_moved"):
        a = summarize(stock_ic(g, "snap_imb1", "ret_30"))
        b = summarize(stock_ic(g, "imb1", "ret_30"))
        rows.append({"MidMovedBetweenTandSnapshot": bool(moved), "Share": len(g) / len(m),
                     "IC_snapshot": a["MeanIC"], "IC_noLookahead": b["MeanIC"]})
    return pd.DataFrame(rows)


def flow_vs_book(d: pd.DataFrame) -> pd.DataFrame:
    sel = json.loads((OUT / "selection_frozen.json").read_text(encoding="utf-8"))
    disc, hold = segment(d, DISCOVERY), segment(d, HOLDOUT).copy()
    z = zscore(hold, sel["book"] + sel["flow"], disc)
    hold["BOOK"] = z[sel["book"]].mean(axis=1)
    hold["FLOW"] = z[sel["flow"]].mean(axis=1)

    def resid(g: pd.DataFrame) -> pd.Series:
        x = g["BOOK"].to_numpy()
        y = g["FLOW"].to_numpy()
        beta = np.dot(x - x.mean(), y - y.mean()) / max(np.dot(x - x.mean(), x - x.mean()), 1e-12)
        return pd.Series(y - beta * x, index=g.index)

    hold["FLOW_resid"] = hold.groupby("Symbol", group_keys=False).apply(resid, include_groups=False)
    corr = hold.groupby("Symbol").apply(lambda g: g.BOOK.rank().corr(g.FLOW.rank()), include_groups=False)
    rows = [{"Item": "Corr(FLOW, BOOK) 逐股票 Spearman 均值", "Value": float(corr.mean())}]
    for h in (10, 30, 60):
        s = summarize(stock_ic(hold, "FLOW_resid", f"ret_{h}"))
        rows.append({"Item": f"FLOW 对 BOOK 残差化后的 IC（{h}s）", "Value": s["MeanIC"]})
        rows.append({"Item": f"  跨股票 t（{h}s）", "Value": s["TStat"]})
    # 探索：按「发现段」IC 定两信号权重（不看下午），再看下午 IC。不计入预注册检验。
    dz = zscore(disc, sel["book"] + sel["flow"], disc)
    disc = disc.assign(BOOK=dz[sel["book"]].mean(axis=1), FLOW=dz[sel["flow"]].mean(axis=1))
    wb = stock_ic(disc, "BOOK", "ret_30").mean()
    wf = stock_ic(disc, "FLOW", "ret_30").mean()
    sd_b = disc.groupby("Symbol").BOOK.std()
    sd_f = disc.groupby("Symbol").FLOW.std()
    hold["ICW"] = wb * hold.BOOK / sd_b.reindex(hold.Symbol).to_numpy() + wf * hold.FLOW / sd_f.reindex(hold.Symbol).to_numpy()
    rows.append({"Item": f"探索：发现段 IC 加权组合权重 BOOK:FLOW = {wb:.3f}:{wf:.3f}", "Value": np.nan})
    for h in (10, 30, 60):
        ic_new = stock_ic(hold, "ICW", f"ret_{h}")
        ic_book = stock_ic(hold, "BOOK", f"ret_{h}")
        diff = (ic_new - ic_book).dropna()
        rows.append({"Item": f"  探索组合 IC（{h}s）", "Value": float(ic_new.mean())})
        rows.append({"Item": f"  − BOOK，配对 t（{h}s）", "Value": float(diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff)))})
    return pd.DataFrame(rows)


def main() -> None:
    d = build_panel()
    gap = float((d.micro - d.imb1 / 2).abs().max())
    t1 = timing_decomposition(d)
    t2 = flow_vs_book(d)
    t1.to_csv(OUT / "posthoc_timing.csv", index=False)
    t2.to_csv(OUT / "posthoc_flow_vs_book.csv", index=False)
    print(f"max |micro − imb1/2| = {gap:.2e}\n")
    print(t1.round(4).to_string(index=False), "\n")
    print(t2.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
