"""
固定时间网格采样 + 逐笔订单流累计量。

采样口径（无前视）：网格时刻 t 的盘口 = 所有 Time < t 的事件回放后的状态。
不使用交易所快照（快照标签 T 的内容对应 T+相位偏移，最多晚 0.95 秒，直接用会前视）。

每个网格点记录：
  - 五档盘口 bid/ask price/volume
  - 累计计数器（窗口特征 = 两个网格点之差，O(1)）：
      aggr_buy_v / aggr_sell_v     主动买 / 主动卖成交量（连续竞价）
      aggr_buy_n / aggr_sell_n     主动买 / 主动卖成交笔数
      big_buy_v / big_sell_v       来自「大单」主动方的成交量（该主动委托累计成交额已 ≥ BIG_NOTIONAL，
                                   按成交发生时已知的累计额判定，无前视）
      add_bid_near / add_ask_near  距本方最优价 NEAR_BPS 以内的新增挂单量
      cxl_bid_near / cxl_ask_near  距本方最优价 NEAR_BPS 以内的撤单量
      ofi                          一档订单流失衡（Cont, Kukanov & Stoikov 2014），逐事件累加
"""

from __future__ import annotations

import heapq

import numpy as np

from .book import BUY, SELL, OrderBook
from .events import EV_ADD, EV_DEL, SymbolEvents
from .replay import to_ms

LEVELS = 5
NEAR_BPS = 10           # 「近端」= 距本方最优价 10bp 以内
BIG_NOTIONAL = 500_000  # 大单主动方：累计成交额 ≥ 50 万元（价格单位为分，比较时 ×100）
COUNTERS = ("aggr_buy_v", "aggr_sell_v", "aggr_buy_n", "aggr_sell_n", "big_buy_v", "big_sell_v",
            "add_bid_near", "add_ask_near", "cxl_bid_near", "cxl_ask_near", "ofi")


class TrackedBook(OrderBook):
    """在 OrderBook 上用惰性堆维护最优价，支持逐事件读取一档。"""
    __slots__ = ("_bh", "_ah")

    def __init__(self) -> None:
        super().__init__()
        self._bh: list[int] = []      # -price
        self._ah: list[int] = []

    def _level_add(self, side: int, price: int, delta: int) -> None:
        book = self.bids if side == BUY else self.asks
        new_level = price not in book
        super()._level_add(side, price, delta)
        if new_level and price in book:
            heapq.heappush(self._bh if side == BUY else self._ah, -price if side == BUY else price)

    def best_bid(self) -> tuple[int, int]:
        h = self._bh
        while h and -h[0] not in self.bids:
            heapq.heappop(h)
        return (-h[0], self.bids[-h[0]]) if h else (0, 0)

    def best_ask(self) -> tuple[int, int]:
        h = self._ah
        while h and h[0] not in self.asks:
            heapq.heappop(h)
        return (h[0], self.asks[h[0]]) if h else (0, 0)


def ofi_increment(pb0, qb0, pa0, qa0, pb1, qb1, pa1, qa1) -> int:
    """单步 OFI：买方贡献 − 卖方贡献。"""
    if pb1 > pb0:
        eb = qb1
    elif pb1 == pb0:
        eb = qb1 - qb0
    else:
        eb = -qb0
    if pa1 < pa0 or pa0 == 0:
        ea = qa1
    elif pa1 == pa0:
        ea = qa1 - qa0
    else:
        ea = -qa0
    return eb - ea


def sample_grid(ev: SymbolEvents, grid_time: np.ndarray, continuous_from: int = 93_000_000,
                continuous_to: int = 145_700_000) -> dict[str, np.ndarray]:
    """在网格时刻（HHMMSSmmm，升序）采样盘口与累计订单流。"""
    g = to_ms(grid_time)
    t = to_ms(ev.time)
    cfrom, cto = int(to_ms(np.array([continuous_from]))[0]), int(to_ms(np.array([continuous_to]))[0])
    ng = len(g)
    out = {f"{s}{k}{lv}": np.zeros(ng, dtype=np.int64)
           for lv in range(1, LEVELS + 1) for s in ("Bid", "Ask") for k in ("Price", "Volume")}
    out.update({c: np.zeros(ng, dtype=np.int64) for c in COUNTERS})

    book = TrackedBook()
    cnt = dict.fromkeys(COUNTERS, 0)
    aggr_notional: dict[int, int] = {}
    kind, side, price, vol, oid, oid2 = (a.tolist() for a in (ev.kind, ev.side, ev.price, ev.volume, ev.oid, ev.oid2))
    tl = t.tolist()
    n = len(kind)
    i = 0
    pb, qb = 0, 0
    pa, qa = 0, 0
    for j in range(ng):
        gj = g[j]
        while i < n and tl[i] < gj:
            k = kind[i]
            live = cfrom <= tl[i] < cto
            if k == EV_ADD:
                p, s = price[i], side[i]
                if live:
                    if s == BUY and pb and p >= pb * (1 - NEAR_BPS / 1e4):
                        cnt["add_bid_near"] += vol[i]
                    elif s == SELL and pa and p <= pa * (1 + NEAR_BPS / 1e4):
                        cnt["add_ask_near"] += vol[i]
                book.add(oid[i], s, p, vol[i])
            elif k == EV_DEL:
                o = book.orders.get(oid[i])
                if live and o is not None:
                    s, p = o[0], o[1]
                    if s == BUY and pb and p >= pb * (1 - NEAR_BPS / 1e4):
                        cnt["cxl_bid_near"] += min(vol[i], o[2])
                    elif s == SELL and pa and p <= pa * (1 + NEAR_BPS / 1e4):
                        cnt["cxl_ask_near"] += min(vol[i], o[2])
                book.reduce(oid[i], vol[i], "cancel")
            else:
                s = side[i]
                if live and s in (BUY, SELL):
                    aggressor = oid[i] if s == BUY else oid2[i]
                    notional = aggr_notional.get(aggressor, 0) + price[i] * vol[i]
                    aggr_notional[aggressor] = notional
                    big = notional >= BIG_NOTIONAL * 100
                    if s == BUY:
                        cnt["aggr_buy_v"] += vol[i]
                        cnt["aggr_buy_n"] += 1
                        if big:
                            cnt["big_buy_v"] += vol[i]
                    else:
                        cnt["aggr_sell_v"] += vol[i]
                        cnt["aggr_sell_n"] += 1
                        if big:
                            cnt["big_sell_v"] += vol[i]
                book.trade(s, oid[i], oid2[i], vol[i])
            nb, nqb = book.best_bid()
            na, nqa = book.best_ask()
            if live and pb and pa and nb and na:
                cnt["ofi"] += ofi_increment(pb, qb, pa, qa, nb, nqb, na, nqa)
            pb, qb, pa, qa = nb, nqb, na, nqa
            i += 1
        bids, asks = book.top(LEVELS)
        for lv in range(LEVELS):
            if lv < len(bids):
                out[f"BidPrice{lv + 1}"][j], out[f"BidVolume{lv + 1}"][j] = bids[lv]
            if lv < len(asks):
                out[f"AskPrice{lv + 1}"][j], out[f"AskVolume{lv + 1}"][j] = asks[lv]
        for c in COUNTERS:
            out[c][j] = cnt[c]
    return out
