#!/usr/bin/env python3
"""报告用图（静态 PNG，供 GitHub README 直接显示）。输出：docs/img/"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs/img"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK TC", "PingFang SC", "Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

LABELS = {
    "time_le": "time_le  事件≤T（参考仓库）",
    "time_sec": "time_sec  事件≤T+999ms",
    "trade": "trade  成交笔数锚定",
    "totals": "totals  成交笔数+总挂单量",
    "phase": "phase  T+固定相位偏移",
    "phase_totals": "phase_totals  相位+总量（推荐）",
}


def recon_policies() -> None:
    s = pd.read_csv(ROOT / "outputs/reconstruction/validation_summary.csv").set_index("Policy").loc[list(LABELS)]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = range(len(s))[::-1]
    ax.barh(list(y), s.Top10AllExact * 100, color=BLUE, height=0.55)
    for yi, v in zip(y, s.Top10AllExact * 100):
        ax.text(v + 1, yi, f"{v:.2f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(list(y), [LABELS[p] for p in s.index])
    ax.set_xlim(0, 112)
    ax.set_xlabel("十档价格与数量全部与交易所快照一致的比例（%）")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("同一套回放引擎，只换对齐规则：236,581 个连续竞价快照", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "reconstruction_policies.png", dpi=150)
    plt.close(fig)


def offsets() -> None:
    o = pd.read_csv(ROOT / "outputs/reconstruction/offsets.csv").sort_values("OffsetMs").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.vlines(o.index, 0, o.OffsetMs, color=BLUE, lw=2)
    ax.scatter(o.index, o.OffsetMs, s=22, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(o.index, o.Symbol.astype(str), rotation=90, fontsize=6.5)
    ax.set_ylabel("快照内容晚于标签时间（ms）")
    ax.set_ylim(0, 1050)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("每只股票的快照都有固定相位偏移（50~990ms），上午估计 = 全天估计（50/50）", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "snapshot_offsets.png", dpi=150)
    plt.close(fig)


def cost_wall() -> None:
    c = pd.read_csv(ROOT / "outputs/strategy/cost_wall.csv")
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    colors = [BLUE if v > 0 else ORANGE for v in c.FwdMid30sBps]
    x = range(len(c))
    ax.bar(x, c.FwdMid30sBps, color=colors, width=0.6)
    half_rt = (c.RoundTripSpreadBps.mean())
    ax.axhline(half_rt, color=INK2, lw=1, ls="--")
    ax.axhline(-half_rt, color=INK2, lw=1, ls="--")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.text(len(c) - 0.6, half_rt + 0.25, f"一次往返价差 ≈ {half_rt:.1f}bp（尚未计佣金）", ha="right", color=INK2, fontsize=9)
    for xi, v in zip(x, c.FwdMid30sBps):
        ax.text(xi, v + (0.25 if v > 0 else -0.25), f"{v:+.2f}", ha="center", va="bottom" if v > 0 else "top", fontsize=8.5, color=INK)
    ax.set_xticks(list(x), c.bucket)
    ax.set_xlabel("下午留出段：ALL 信号分位")
    ax.set_ylabel("未来 30 秒中间价变动（bp）")
    ax.set_ylim(-half_rt - 1.5, half_rt + 1.5)
    ax.set_title("信号单调有效，但最强 1% 只预测 ±1.4bp，跨一次价差就要 ~7bp", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "cost_wall.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    recon_policies()
    offsets()
    cost_wall()
    print("charts ->", IMG)
