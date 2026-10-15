"""订单流特征的单元测试。"""

import numpy as np

from lob.book import BUY, SELL
from lob.events import EV_ADD, EV_DEL, EV_TRADE, SymbolEvents
from lob.flow import TrackedBook, ofi_increment, sample_grid


def test_ofi_increment_cases():
    # 买一价上移贡献新买一量 +3；卖一价上移贡献 −(−旧卖一量) = +5 → 两边都偏多
    assert ofi_increment(100, 5, 101, 5, 101, 3, 102, 5) == 3 + 5
    assert ofi_increment(100, 5, 101, 5, 100, 8, 101, 5) == 3          # 买一加量
    assert ofi_increment(100, 5, 101, 5, 100, 5, 101, 2) == 3          # 卖一被吃
    assert ofi_increment(100, 5, 101, 5, 99, 4, 101, 5) == -5          # 买一被打穿


def test_tracked_book_best_after_level_removed():
    b = TrackedBook()
    b.add(1, BUY, 100, 10)
    b.add(2, BUY, 101, 10)
    b.add(3, SELL, 103, 10)
    assert b.best_bid() == (101, 10)
    b.reduce(2, 10, "cancel")
    assert b.best_bid() == (100, 10)
    b.add(4, BUY, 101, 7)                    # 价位重新出现
    assert b.best_bid() == (101, 7) and b.best_ask() == (103, 10)


def _ev(rows):
    a = np.array(rows, dtype=np.int64).T
    return SymbolEvents(600000, a[0], a[1], a[2].astype(np.int8), a[3].astype(np.int8), a[4], a[5], a[6], a[7], [])


def test_sample_grid_no_lookahead_and_counters():
    ev = _ev([
        (1, 93000100, EV_ADD, BUY, 1000, 500, 1, 0),
        (2, 93000200, EV_ADD, SELL, 1001, 500, 2, 0),
        (3, 93001000, EV_TRADE, BUY, 1001, 200, 99, 2),   # 恰在网格点 93001000：不应计入该点
        (4, 93001500, EV_DEL, BUY, 1000, 100, 1, 0),
    ])
    out = sample_grid(ev, np.array([93001000, 93002000]))
    assert out["AskVolume1"][0] == 500 and out["aggr_buy_v"][0] == 0
    assert out["AskVolume1"][1] == 300 and out["aggr_buy_v"][1] == 200
    assert out["cxl_bid_near"][1] == 100
    assert out["ofi"][1] == 200 - 100        # 卖一被吃 200（+200），买一撤 100（−100）
