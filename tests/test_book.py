"""盘口状态机与对齐策略的单元测试（合成数据，不依赖原始行情）。"""

import numpy as np
import pytest

from lob.book import BUY, SELL, OrderBook
from lob.events import EV_ADD, EV_DEL, EV_TRADE, SymbolEvents
from lob.replay import cut_points, replay, totals_path


def make_events(rows):
    """rows: (biz, time, kind, side, price, volume, oid, oid2)"""
    a = np.array(rows, dtype=np.int64).T
    return SymbolEvents(600000, a[0], a[1], a[2].astype(np.int8), a[3].astype(np.int8),
                        a[4], a[5], a[6], a[7], status=[])


# ---------------------------------------------------------------- OrderBook
def test_add_and_levels_aggregate():
    b = OrderBook()
    b.add(1, BUY, 1000, 300)
    b.add(2, BUY, 1000, 200)
    b.add(3, SELL, 1001, 100)
    bids, asks = b.top(10)
    assert bids == [(1000, 500)] and asks == [(1001, 100)]
    assert (b.total_bid, b.total_ask) == (500, 100)


def test_partial_cancel_then_full_cancel_removes_level():
    b = OrderBook()
    b.add(1, BUY, 1000, 300)
    b.reduce(1, 100, "cancel")
    assert b.bids == {1000: 200}
    b.reduce(1, 200, "cancel")
    assert b.bids == {} and 1 not in b.orders and b.total_bid == 0


def test_continuous_trade_reduces_only_passive_side():
    b = OrderBook()
    b.add(10, SELL, 1001, 500)              # 被动卖
    b.trade(BUY, bid_oid=99, ask_oid=10, volume=200)   # 主动买 99 不在盘口
    assert b.asks == {1001: 300}
    assert b.anomalies["trade_passive_ask_missing"] == 0


def test_auction_trade_reduces_both_sides():
    b = OrderBook()
    b.add(1, BUY, 1002, 300)
    b.add(2, SELL, 1000, 200)
    b.trade(0, bid_oid=1, ask_oid=2, volume=200)
    assert b.bids == {1002: 100} and b.asks == {}


def test_missing_and_overfill_are_counted_not_raised():
    b = OrderBook()
    b.reduce(7, 100, "cancel")
    b.add(1, SELL, 1001, 100)
    b.trade(BUY, 0, 1, 150)
    assert b.anomalies["cancel_missing"] == 1
    assert b.anomalies["trade_passive_ask_overfill"] == 1
    assert b.asks == {}


def test_top_orders_prices():
    b = OrderBook()
    for i, p in enumerate([995, 1000, 998, 1003]):
        b.add(i, BUY, p, 100)
    for i, p in enumerate([1010, 1005, 1007], start=10):
        b.add(i, SELL, p, 100)
    bids, asks = b.top(2)
    assert [p for p, _ in bids] == [1003, 1000]
    assert [p for p, _ in asks] == [1005, 1007]


# ---------------------------------------------------------------- 对齐
@pytest.fixture
def stream():
    # 时间 93002 秒内发生：一笔卖单挂出 → 主动买吃掉一部分 → 一笔买单挂出 → 撤掉 → 下一秒再成交
    return make_events([
        (1, 93001500, EV_ADD,   SELL, 1001, 500, 10, 0),   # 0
        (2, 93002100, EV_TRADE, BUY,  1001, 200, 99, 10),  # 1  第 1 笔成交
        (3, 93002400, EV_ADD,   BUY,  1000, 300, 20, 0),   # 2
        (4, 93002800, EV_DEL,   BUY,  1000, 300, 20, 0),   # 3
        (5, 93003200, EV_TRADE, BUY,  1001, 100, 98, 10),  # 4  第 2 笔成交
    ])


def test_time_le_misses_same_second_events(stream):
    # 快照 Time=93002000，真实状态包含 93002xxx 的事件
    cuts, _ = cut_points(stream, np.array([93002000]), np.array([1]),
                         np.array([0]), np.array([300]), "time_le")
    assert cuts[0] == 1                      # 只应用了挂单，漏掉第 1 笔成交


def test_time_sec_includes_whole_second(stream):
    cuts, _ = cut_points(stream, np.array([93002000]), np.array([1]),
                         np.array([0]), np.array([300]), "time_sec")
    assert cuts[0] == 4


def test_trade_anchor_never_crosses_next_trade(stream):
    # MatchItem=1：切点必须在第 1 笔成交之后（≥2）、第 2 笔之前（≤4）
    cuts, _ = cut_points(stream, np.array([93003000]), np.array([1]),
                         np.array([0]), np.array([300]), "trade")
    assert 2 <= cuts[0] <= 4


def test_totals_anchor_picks_matching_gap_position(stream):
    # 快照显示买方还有 300（撤单前）：缺口 [2,4] 内只有切点 3 满足
    path = totals_path(stream)
    cuts, how = cut_points(stream, np.array([93002000]), np.array([1]),
                           np.array([300]), np.array([300]), "totals", path)
    assert cuts[0] == 3 and how[0] == 1
    out, _ = replay(stream, cuts, levels=1)
    assert list(out[0]) == [1000, 300, 1001, 300]


def test_replay_handles_unsorted_cuts(stream):
    out, book = replay(stream, np.array([5, 1, 3]), levels=1)
    assert list(out[1]) == [0, 0, 1001, 500]
    assert list(out[2]) == [1000, 300, 1001, 300]
    assert list(out[0]) == [0, 0, 1001, 200]
    assert book.asks == {1001: 200}


def test_to_ms_crosses_minute_boundary():
    from lob.replay import to_ms
    assert to_ms(np.array([93059990]))[0] + 10 == to_ms(np.array([93100000]))[0]


def test_phase_cut_uses_fixed_offset(stream):
    # 偏移 450ms：快照 T=93002000 对应 93002.450 那一刻 → 应用 93001500/93002100/93002400 三个事件
    cuts, _ = cut_points(stream, np.array([93002000]), np.array([1]),
                         np.array([0]), np.array([0]), "phase", offset_ms=450)
    assert cuts[0] == 3


def test_estimate_offset_recovers_true_phase(stream):
    from lob.replay import estimate_offset
    path = totals_path(stream)
    # 快照内容 = 切点 3（买 300 / 卖 300）：约束 93002400 ≤ T+off ≤ 93002800
    off, share = estimate_offset(stream, np.array([93002000]), np.array([1]),
                                 np.array([300]), np.array([300]), path)
    assert 400 <= off <= 800 and share == 1.0
