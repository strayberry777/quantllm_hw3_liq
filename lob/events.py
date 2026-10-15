"""
事件解码：把逐笔委托 ord 与逐笔成交 exe 合并成按 BizIndex 排序的统一事件流。

编码（与参考仓库一致，经本仓库数据核对）：
    ord.OrderKind   65 'A' 新增（Volume = 进入盘口的剩余量） / 68 'D' 撤单 / 83 'S' 产品状态
    ord.FunctionCode 66 'B' 买 / 83 'S' 卖；状态消息里是 C/I/J/O/V 等阶段码
    exe.BSFlag      66 主动买 / 83 主动卖 / 0 集合竞价（无主动方）

排序键只用 BizIndex：它在委托与成交两张表之间全局唯一、严格递增（本数据 0 重复），
本身就是交易所的处理顺序。参考仓库用 (Time, BizIndex, EventType) 排序，结果相同，
但 Time 是 10ms 精度，单用 Time 会把同一 10ms 内的事件顺序弄乱。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

ADD, DELETE, STATUS = 65, 68, 83
BUY, SELL = 66, 83

# 统一事件类型
EV_ADD, EV_DEL, EV_TRADE = 0, 1, 2

REGULAR_CLOSE = 150_000_000          # 15:00:00.000；之后的成交是科创板盘后固定价格交易


@dataclass
class SymbolEvents:
    symbol: int
    biz: np.ndarray        # int64  BizIndex
    time: np.ndarray       # int64  HHMMSSmmm
    kind: np.ndarray       # int8   EV_ADD / EV_DEL / EV_TRADE
    side: np.ndarray       # int8   委托方向（ADD/DEL）；成交时为 BSFlag
    price: np.ndarray      # int64  价格（分）
    volume: np.ndarray     # int64
    oid: np.ndarray        # int64  委托：OrderOriNo；成交：BidOrder
    oid2: np.ndarray       # int64  成交：AskOrder；其余 0
    status: list[tuple[int, int, int]]   # (BizIndex, Time, 阶段码) 产品状态消息

    def __len__(self) -> int:
        return len(self.biz)

    @property
    def trade_positions(self) -> np.ndarray:
        """第 k 笔成交（从 1 计）在事件流中的下标，用于按 MatchItem 对齐。"""
        return np.flatnonzero(self.kind == EV_TRADE)


def load_tables(data_dir: Path, date: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ord_ = pl.read_parquet(data_dir / f"ord_{date}.parquet")
    exe = pl.read_parquet(data_dir / f"exe_{date}.parquet")
    snp = pl.read_parquet(data_dir / f"snp_{date}.parquet")
    return ord_, exe, snp


def build_events(ord_sym: pl.DataFrame, exe_sym: pl.DataFrame, symbol: int) -> SymbolEvents:
    """单只股票的统一事件流。盘后（15:00 之后）成交单独成市，不进入连续竞价盘口。"""
    status = (
        ord_sym.filter(pl.col("OrderKind") == STATUS)
        .select("BizIndex", "Time", "FunctionCode").rows()
    )
    orders = ord_sym.filter(pl.col("OrderKind").is_in([ADD, DELETE])).select(
        pl.col("BizIndex").cast(pl.Int64).alias("biz"),
        pl.col("Time").cast(pl.Int64).alias("time"),
        pl.when(pl.col("OrderKind") == ADD).then(EV_ADD).otherwise(EV_DEL).cast(pl.Int8).alias("kind"),
        pl.col("FunctionCode").cast(pl.Int8).alias("side"),
        (pl.col("Price").cast(pl.Float64) * 100).round().cast(pl.Int64).alias("price"),
        pl.col("Volume").cast(pl.Int64).alias("volume"),
        pl.col("OrderOriNo").cast(pl.Int64).alias("oid"),
        pl.lit(0, pl.Int64).alias("oid2"),
    )
    trades = exe_sym.filter(pl.col("Time") <= REGULAR_CLOSE).select(
        pl.col("BizIndex").cast(pl.Int64).alias("biz"),
        pl.col("Time").cast(pl.Int64).alias("time"),
        pl.lit(EV_TRADE, pl.Int8).alias("kind"),
        pl.col("BSFlag").cast(pl.Int8).alias("side"),
        (pl.col("Price").cast(pl.Float64) * 100).round().cast(pl.Int64).alias("price"),
        pl.col("Volume").cast(pl.Int64).alias("volume"),
        pl.col("BidOrder").cast(pl.Int64).alias("oid"),
        pl.col("AskOrder").cast(pl.Int64).alias("oid2"),
    )
    ev = pl.concat([orders, trades]).sort("biz")
    if ev["biz"].is_duplicated().any():
        raise ValueError(f"{symbol}: BizIndex 重复，排序键不唯一")
    return SymbolEvents(
        symbol=symbol,
        **{c: ev[c].to_numpy() for c in ("biz", "time", "kind", "side", "price", "volume", "oid", "oid2")},
        status=status,
    )
