"""
回放与对齐。

核心思想：「重建」与「对齐」分开。
- replay(events, cuts)：按顺序回放事件，在每个切点 cuts[i]（= 已应用的事件个数）输出十档。
  只用逐笔流构造盘口，从不读快照的档位。
- cut_points()：决定每个快照应该切在第几个事件。不同对齐策略只在这里不同。

对齐策略（逐级加入快照里的「对时信息」，都不使用快照的十档明细）：
  time_le       事件 Time ≤ 快照 Time                            —— 参考仓库的规则
  time_sec      事件 Time ≤ 快照 Time + 999ms                     —— 假设快照时间是秒级截断
  trade         切在第 MatchItem 笔成交之后、下一笔成交之前；缺口内按 time_sec 取舍
  totals        trade 缺口内，选使 (全部买量, 全部卖量) = 快照 (TotalBidVolume, TotalAskVolume) 的切点
  phase         事件 Time < 快照 Time + 该股固定相位偏移，再夹到 trade 缺口内（只用时间）
  phase_totals  totals 的候选里，取离 phase 切点最近的一个（本仓库推荐）

相位偏移：每只股票的快照内容对应「标签时间 T + 固定毫秒数」那一刻（实测 340~950ms 不等）。
estimate_offset() 只用总量匹配（TotalBid/AskVolume）且无歧义的快照来估计，不读十档。
"""

from __future__ import annotations

import numpy as np

from .book import OrderBook
from .events import EV_ADD, EV_DEL, SymbolEvents

POLICIES = ("time_le", "time_sec", "trade", "totals", "phase", "phase_totals")


def to_ms(t: np.ndarray) -> np.ndarray:
    """HHMMSSmmm → 当日毫秒数。"""
    t = np.asarray(t, dtype=np.int64)
    return ((t // 10_000_000) * 3600 + (t // 100_000) % 100 * 60 + (t // 1000) % 100) * 1000 + t % 1000


def apply(book: OrderBook, ev: SymbolEvents, i: int) -> None:
    k = ev.kind[i]
    if k == EV_ADD:
        book.add(int(ev.oid[i]), int(ev.side[i]), int(ev.price[i]), int(ev.volume[i]))
    elif k == EV_DEL:
        book.reduce(int(ev.oid[i]), int(ev.volume[i]), "cancel")
    else:
        book.trade(int(ev.side[i]), int(ev.oid[i]), int(ev.oid2[i]), int(ev.volume[i]))


def totals_path(ev: SymbolEvents) -> tuple[np.ndarray, np.ndarray]:
    """第一遍回放：应用前 c 个事件后的 (全部买量, 全部卖量)，c = 0..n。"""
    n = len(ev)
    tb = np.zeros(n + 1, dtype=np.int64)
    ta = np.zeros(n + 1, dtype=np.int64)
    book = OrderBook()
    kind, side, price, vol, oid, oid2 = (a.tolist() for a in (ev.kind, ev.side, ev.price, ev.volume, ev.oid, ev.oid2))
    for i in range(n):
        k = kind[i]
        if k == EV_ADD:
            book.add(oid[i], side[i], price[i], vol[i])
        elif k == EV_DEL:
            book.reduce(oid[i], vol[i], "cancel")
        else:
            book.trade(side[i], oid[i], oid2[i], vol[i])
        tb[i + 1] = book.total_bid
        ta[i + 1] = book.total_ask
    return tb, ta


def trade_gap(ev: SymbolEvents, match_item: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """第 MatchItem 笔成交之后 ~ 第 MatchItem+1 笔成交之前 的切点区间 [lo, hi]。
    MatchItem=0（开盘前）时区间为 [0, 第 1 笔成交]。"""
    tp = ev.trade_positions
    n = len(ev)
    k = match_item.astype(np.int64)
    first = tp[0] if len(tp) else n
    lo = np.where(k > 0, tp[np.clip(k - 1, 0, max(len(tp) - 1, 0))] + 1, 0)
    hi = np.where(k < len(tp), tp[np.clip(k, 0, max(len(tp) - 1, 0))], n)
    hi = np.where(k > 0, hi, first)
    return lo, hi


def totals_candidates(path, lo, hi, total_bid, total_ask, j) -> np.ndarray:
    tb, ta = path
    a, b = int(lo[j]), int(hi[j])
    return a + np.flatnonzero((tb[a:b + 1] == total_bid[j]) & (ta[a:b + 1] == total_ask[j]))


def estimate_offset(ev: SymbolEvents, snap_time: np.ndarray, match_item: np.ndarray,
                    total_bid: np.ndarray, total_ask: np.ndarray, path, mask: np.ndarray | None = None,
                    grid=range(0, 3000, 10)) -> tuple[int, float]:
    """用「总量唯一匹配」的快照估计固定相位偏移（毫秒）。

    每个唯一候选切点 c 给出约束：第 c-1 个事件时间 ≤ T+off ≤ 第 c 个事件时间。
    在 10ms 网格上取满足约束最多的 off。返回 (off, 满足率)。
    """
    t = to_ms(ev.time)
    T = to_ms(snap_time)
    lo, hi = trade_gap(ev, match_item)
    n = len(ev)
    lows, highs = [], []
    idx = range(len(snap_time)) if mask is None else np.flatnonzero(mask)
    for j in idx:
        cand = totals_candidates(path, lo, hi, total_bid, total_ask, j)
        if len(cand) != 1:
            continue
        c = int(cand[0])
        lows.append((t[c - 1] if c > 0 else -10**9) - T[j])
        highs.append((t[c] if c < n else 10**9) - T[j])
    if not lows:
        return 0, 0.0
    lows, highs = np.array(lows), np.array(highs)
    best, share = 0, -1.0
    for off in grid:
        s = float(np.mean((lows <= off) & (off <= highs)))
        if s > share:
            best, share = off, s
    return best, share


def cut_points(ev: SymbolEvents, snap_time: np.ndarray, match_item: np.ndarray,
               total_bid: np.ndarray, total_ask: np.ndarray, policy: str,
               path=None, offset_ms: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """返回 (cuts, how)。how：totals 类策略 1 = 总量命中，0 = 回退到时间规则。"""
    if policy not in POLICIES:
        raise ValueError(policy)
    t = to_ms(ev.time)
    T = to_ms(snap_time)
    how = np.zeros(len(snap_time), dtype=np.int8)
    if policy == "time_le":
        return np.searchsorted(t, T, side="right"), how
    sec_cut = np.searchsorted(t, T + 999, side="right")
    if policy == "time_sec":
        return sec_cut, how

    if policy.startswith("phase"):
        if offset_ms is None:
            raise ValueError("phase 策略需要 offset_ms")
        base = np.searchsorted(t, T + offset_ms, side="left")      # 严格早于 T+off 的事件
    else:
        base = sec_cut
    lo, hi = trade_gap(ev, match_item)
    cuts = np.clip(base, lo, hi)
    if policy in ("trade", "phase"):
        return cuts, how

    if path is None:
        path = totals_path(ev)
    for j in range(len(snap_time)):
        cand = totals_candidates(path, lo, hi, total_bid, total_ask, j)
        if len(cand):
            cuts[j] = cand[np.argmin(np.abs(cand - cuts[j]))]
            how[j] = 1
    return cuts, how


def replay(ev: SymbolEvents, cuts: np.ndarray, levels: int = 10) -> tuple[np.ndarray, OrderBook]:
    """按切点输出盘口：shape (len(cuts), levels*4)，列序 [BidP1, BidV1, AskP1, AskV1, BidP2, ...]，价格单位为分。"""
    order = np.argsort(cuts, kind="stable")
    out = np.zeros((len(cuts), levels * 4), dtype=np.int64)
    book = OrderBook()
    kind, side, price, vol, oid, oid2 = (a.tolist() for a in (ev.kind, ev.side, ev.price, ev.volume, ev.oid, ev.oid2))
    i = 0
    for j in order:
        c = int(cuts[j])
        while i < c:
            k = kind[i]
            if k == EV_ADD:
                book.add(oid[i], side[i], price[i], vol[i])
            elif k == EV_DEL:
                book.reduce(oid[i], vol[i], "cancel")
            else:
                book.trade(side[i], oid[i], oid2[i], vol[i])
            i += 1
        bids, asks = book.top(levels)
        row = out[j]
        for lv in range(levels):
            if lv < len(bids):
                row[lv * 4], row[lv * 4 + 1] = bids[lv]
            if lv < len(asks):
                row[lv * 4 + 2], row[lv * 4 + 3] = asks[lv]
    while i < len(ev):
        apply(book, ev, i)
        i += 1
    return out, book
