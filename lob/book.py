"""
订单簿状态机：维护每笔活动委托的剩余量与各价位聚合量。

规则：
- ADD：以 OrderOriNo 为键登记（上交所委托流里只出现「进入盘口的剩余部分」，
  即时全部成交的主动单不会出现 ADD；部分成交的主动单在其成交之后出现，Volume 为剩余量）
- DEL：按撤单量扣减
- 成交：主动买扣减被动卖单（AskOrder），主动卖扣减被动买单（BidOrder）；
  集合竞价成交（BSFlag=0）双方都是盘口里的挂单，两边都扣减
- 扣减找不到委托或量不足时不报错，但计数（anomalies），用于误差归因
"""

from __future__ import annotations

import heapq
from collections import Counter

BUY, SELL = 66, 83


class OrderBook:
    __slots__ = ("orders", "bids", "asks", "total_bid", "total_ask", "anomalies")

    def __init__(self) -> None:
        self.orders: dict[int, list] = {}         # oid -> [side, price, remaining]
        self.bids: dict[int, int] = {}            # price -> volume
        self.asks: dict[int, int] = {}
        self.total_bid = 0                        # 全部买单剩余量之和（对应快照 TotalBidVolume）
        self.total_ask = 0
        self.anomalies: Counter = Counter()

    # ------------------------------------------------------------------ 基本操作
    def _level_add(self, side: int, price: int, delta: int) -> None:
        book = self.bids if side == BUY else self.asks
        v = book.get(price, 0) + delta
        if v > 0:
            book[price] = v
        else:
            book.pop(price, None)
        if side == BUY:
            self.total_bid += delta
        else:
            self.total_ask += delta

    def add(self, oid: int, side: int, price: int, volume: int) -> None:
        if volume <= 0:
            self.anomalies["add_nonpositive"] += 1
            return
        if oid in self.orders:
            self.anomalies["add_duplicate"] += 1
            self.reduce(oid, self.orders[oid][2], reason="replace")
        self.orders[oid] = [side, price, volume]
        self._level_add(side, price, volume)

    def reduce(self, oid: int, volume: int, reason: str) -> int:
        o = self.orders.get(oid)
        if o is None:
            self.anomalies[f"{reason}_missing"] += 1
            return 0
        side, price, remaining = o
        cut = min(remaining, volume)
        if cut < volume:
            self.anomalies[f"{reason}_overfill"] += 1
        self._level_add(side, price, -cut)
        if remaining - cut > 0:
            o[2] = remaining - cut
        else:
            del self.orders[oid]
        return cut

    def trade(self, bsflag: int, bid_oid: int, ask_oid: int, volume: int) -> None:
        if bsflag == BUY:
            self.reduce(ask_oid, volume, "trade_passive_ask")
        elif bsflag == SELL:
            self.reduce(bid_oid, volume, "trade_passive_bid")
        else:
            self.reduce(bid_oid, volume, "auction_bid")
            self.reduce(ask_oid, volume, "auction_ask")

    # ------------------------------------------------------------------ 读取
    def top(self, n: int = 10) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """买 n 档（价高到低）、卖 n 档（价低到高），每档 (价格分, 数量)。"""
        bp = heapq.nlargest(n, self.bids)
        ap = heapq.nsmallest(n, self.asks)
        return [(p, self.bids[p]) for p in bp], [(p, self.asks[p]) for p in ap]

    def crossed(self) -> bool:
        return bool(self.bids) and bool(self.asks) and max(self.bids) >= min(self.asks)
