#!/usr/bin/env python3
"""
留出段（下午）T+1 策略回测 + 成本墙分析（按 research/PROTOCOL.md §10）。

信号：预注册的 ALL 分数（z 分数用发现段参数）；开仓阈值 = 发现段 ALL 的 99% / 1% 分位（全股票池化）。
账户：沿用参考仓库规则，另加 1 秒执行延迟；--lot 可改每笔股数做规模敏感性。

用法：python scripts/strategy.py [--lot 100]
输出：outputs/strategy/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from factor_research import DISCOVERY, HOLDOUT, OUT as RESEARCH, build_panel, segment, zscore  # noqa: E402

OUT = ROOT / "outputs/strategy"
CAPITAL = 1_000_000
INVESTED = 0.5
COOLDOWN = 600
MAX_TRADES = 2
LATENCY = 1                     # 秒
TRADE_END = 145_500_000


def fee(notional: float, side: str) -> float:
    commission = max(5.0, notional * 0.0003)
    return commission + notional * 0.00001 + (notional * 0.0005 if side == "SELL" else 0.0)


def prepare() -> tuple[pd.DataFrame, float, float]:
    sel = json.loads((RESEARCH / "selection_frozen.json").read_text(encoding="utf-8"))
    cols = sel["book"] + sel["flow"]
    d = build_panel()
    disc = segment(d, DISCOVERY)
    hold = d[(d.Time >= HOLDOUT[0]) & (d.Time <= 145_700_000)].copy()
    zd = zscore(disc, cols, disc)
    lo, hi = np.quantile(zd.mean(axis=1), [0.01, 0.99])
    hold["score"] = zscore(hold, cols, disc).mean(axis=1)
    return hold, float(lo), float(hi)


def backtest(hold: pd.DataFrame, lo: float, hi: float, lot: int) -> tuple[pd.DataFrame, dict]:
    hold = hold.sort_values(["Symbol", "Time"])
    g = hold.groupby("Symbol", sort=False)
    # 延迟成交：用 t+LATENCY 秒的盘口
    for c in ("BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1"):
        hold[f"exec_{c}"] = g[c].shift(-LATENCY)
    first = hold[hold.Time == HOLDOUT[0]].set_index("Symbol")
    last = hold[hold.Time == 145_700_000].set_index("Symbol")
    syms = sorted(first.index)
    alloc = CAPITAL * INVESTED / len(syms)
    pos, sellable, init = {}, {}, {}
    for s in syms:
        mid = first.loc[s, "mid"] / 100
        q = math.floor(alloc / mid / 100) * 100
        pos[s] = sellable[s] = init[s] = q
    cash0 = CAPITAL - sum(init[s] * first.loc[s, "mid"] / 100 for s in syms)
    cash = cash0
    last_t, n_tr = defaultdict(lambda: -10**9), defaultdict(int)
    trades = []
    ev = hold[(hold.Time <= TRADE_END) & ((hold.score >= hi) | (hold.score <= lo))].sort_values(["Time", "Symbol"])
    for r in ev.itertuples(index=False):
        s = r.Symbol
        sec = (r.Time // 10_000_000) * 3600 + (r.Time // 100_000 % 100) * 60 + (r.Time // 1000 % 100)
        if sec - last_t[s] < COOLDOWN or n_tr[s] >= MAX_TRADES or np.isnan(r.exec_AskPrice1):
            continue
        if r.score >= hi:
            px = r.exec_AskPrice1 / 100
            notional = px * lot
            f = fee(notional, "BUY")
            if r.exec_AskVolume1 >= lot and cash >= notional + f and (pos[s] + lot) * px <= CAPITAL * 0.025:
                cash -= notional + f
                pos[s] += lot
                trades.append((r.Time, s, "BUY", px, lot, r.score, f))
            else:
                continue
        else:
            px = r.exec_BidPrice1 / 100
            notional = px * lot
            f = fee(notional, "SELL")
            if r.exec_BidVolume1 >= lot and sellable[s] >= lot:
                cash += notional - f
                pos[s] -= lot
                sellable[s] -= lot
                trades.append((r.Time, s, "SELL", px, lot, r.score, f))
            else:
                continue
        last_t[s] = sec
        n_tr[s] += 1
    t = pd.DataFrame(trades, columns=["Time", "Symbol", "Side", "Price", "Quantity", "Score", "Fee"])
    nav = cash + sum(pos[s] * last.loc[s, "mid"] / 100 for s in syms)
    bench = cash0 + sum(init[s] * last.loc[s, "mid"] / 100 for s in syms)
    fees = float(t.Fee.sum()) if len(t) else 0.0
    summary = {"Lot": lot, "Trades": len(t), "Buys": int((t.Side == "BUY").sum()) if len(t) else 0,
               "Sells": int((t.Side == "SELL").sum()) if len(t) else 0, "Fees": fees,
               "StrategyNAV": nav, "BenchmarkNAV": bench, "ExcessYuan": nav - bench,
               "ExcessBeforeFeesYuan": nav - bench + fees, "ExcessPct": (nav - bench) / CAPITAL,
               "ThresholdLow": lo, "ThresholdHigh": hi}
    return t, summary


def cost_wall(hold: pd.DataFrame) -> pd.DataFrame:
    """按信号分位分组：未来 30s 中间价变动（bp） vs 一次往返成本（bp）。"""
    h = hold[(hold.Time >= HOLDOUT[0]) & (hold.Time <= HOLDOUT[1])].dropna(subset=["ret_30"]).copy()
    h["bucket"] = pd.cut(h.score.rank(pct=True), [0, .01, .05, .2, .5, .8, .95, .99, 1.0],
                         labels=["≤1%", "1–5%", "5–20%", "20–50%", "50–80%", "80–95%", "95–99%", "≥99%"])
    h["half_spread_bps"] = (h.AskPrice1 - h.BidPrice1) / 2 / h.mid * 1e4
    h["px"] = h.mid / 100
    h["fee_rt_100sh_bps"] = (h.px.apply(lambda p: fee(p * 100, "BUY") + fee(p * 100, "SELL")) / (h.px * 100)) * 1e4
    h["fee_rt_1m_bps"] = (fee(1e6, "BUY") + fee(1e6, "SELL")) / 1e6 * 1e4
    out = h.groupby("bucket", observed=True).agg(
        Rows=("ret_30", "size"), FwdMid30sBps=("ret_30", lambda x: x.mean() * 1e4),
        HalfSpreadBps=("half_spread_bps", "mean"))
    out["RoundTripSpreadBps"] = 2 * out.HalfSpreadBps
    out["FeesRoundTrip100SharesBps"] = h.fee_rt_100sh_bps.mean()
    out["FeesRoundTrip1MYuanBps"] = h.fee_rt_1m_bps.mean()
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lots", type=int, nargs="+", default=[100, 1000])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hold, lo, hi = prepare()
    rows = []
    for lot in args.lots:
        t, s = backtest(hold, lo, hi, lot)
        t.to_csv(OUT / f"trades_lot{lot}.csv", index=False)
        rows.append(s)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False)
    cw = cost_wall(hold)
    cw.to_csv(OUT / "cost_wall.csv", index=False)
    print(summ.round(2).T.to_string())
    print(cw.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
