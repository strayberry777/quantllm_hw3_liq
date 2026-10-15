"""重建盘口 vs 交易所快照的逐档比对。"""

from __future__ import annotations

import numpy as np
import polars as pl

LEVELS = 10
SNAP_LEVEL_COLS = [c for lv in range(1, LEVELS + 1)
                   for c in (f"BidPrice{lv}", f"BidVolume{lv}", f"AskPrice{lv}", f"AskVolume{lv}")]

# 校验窗口：连续竞价。14:57 之后是收盘集合竞价，快照显示的是虚拟撮合价，不可比
CONTINUOUS = ((93_000_000, 113_000_000), (130_000_000, 145_700_000))


def snapshot_matrix(snp_sym: pl.DataFrame) -> np.ndarray:
    m = snp_sym.select(SNAP_LEVEL_COLS).to_numpy().astype(np.float64)
    m[:, 0::2] = np.round(m[:, 0::2] * 100)     # 价格 → 分
    return np.round(m).astype(np.int64)


def in_continuous(t: np.ndarray, windows=CONTINUOUS) -> np.ndarray:
    mask = np.zeros(len(t), dtype=bool)
    for a, b in windows:
        mask |= (t >= a) & (t < b)
    return mask


def compare(rebuilt: np.ndarray, source: np.ndarray) -> dict[str, np.ndarray]:
    """逐快照比较。返回每行的布尔/计数数组。"""
    r = rebuilt.reshape(len(rebuilt), LEVELS, 4)
    s = source.reshape(len(source), LEVELS, 4)
    price_eq = (r[:, :, 0] == s[:, :, 0]) & (r[:, :, 2] == s[:, :, 2])
    full_eq = price_eq & (r[:, :, 1] == s[:, :, 1]) & (r[:, :, 3] == s[:, :, 3])
    return {
        "l1_price": price_eq[:, 0],
        "l1_full": full_eq[:, 0],
        "levels_price": price_eq.sum(axis=1),
        "levels_full": full_eq.sum(axis=1),
        "all10_full": full_eq.all(axis=1),
    }
