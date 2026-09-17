#!/usr/bin/env python3
"""Generate Section 5 supporting figures (algorithm/encoding/kernel diagrams)."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MISSILES, TARGET_CENTER, missile_spec, MISSILE_SPEED
from src.kinematics import missile_position
from src.geometry import dist_to_segment


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_interval_refine(out: Path) -> None:
    spec = missile_spec("M1")
    burst = np.array([17188.0, 0.0, 1736.5])
    te = 5.1
    tt = np.linspace(0.0, 20.0, 801)
    p = spec.p0[None, :] + MISSILE_SPEED * spec.direction[None, :] * tt[:, None]
    c = burst[None, :] + np.array([0.0, 0.0, -3.0]) * (tt - te)[:, None]
    d = dist_to_segment(c, p, TARGET_CENTER)
    coarse = np.arange(0.0, 20.01, 0.1)
    pc = spec.p0[None, :] + MISSILE_SPEED * spec.direction[None, :] * coarse[:, None]
    cc = burst[None, :] + np.array([0.0, 0.0, -3.0]) * (coarse - te)[:, None]
    dc = dist_to_segment(cc, pc, TARGET_CENTER)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(tt, d, color="#1f77b4", lw=1.4, label=r"$d(t)$ 视线—云团距离")
    ax.axhline(10.0, color="#d62728", ls="--", lw=1.2, label=r"阈值 $R=10$ m")
    ax.axvspan(8.0379, 9.4481, color="#2ca02c", alpha=0.18, label="精化遮蔽区间")
    active = dc <= 10.0
    ax.plot(coarse[~active], dc[~active], "o", ms=4, color="#999999", label="粗采样（不遮蔽）")
    ax.plot(coarse[active], dc[active], "o", ms=4, color="#ff7f0e", label="粗采样（遮蔽）")
    ax.annotate("brentq 精化左端点", xy=(8.04, 10.0), xytext=(2.2, 26),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
    ax.annotate("brentq 精化右端点", xy=(9.45, 10.0), xytext=(11.0, 22),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
    ax.set_xlabel("时间 $t$ (s)")
    ax.set_ylabel("距离 (m)")
    ax.set_title("粗采样检测 + brentq 二分精化（问题 1）", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, out)


def fig_encoding(out: Path) -> None:
    u = np.array([0.000386, 2.718877, 3.566399])
    td = u + np.array([0.0, 1.0, 2.0])
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    y = 1.0
    ax.axhline(y, color="#cccccc", lw=1)
    for i, (uu, tt) in enumerate(zip(u, td)):
        ax.plot([uu], [y + 0.18], "o", ms=9, mfc="white", mec="#1f77b4", mew=1.5)
        ax.plot([tt], [y - 0.18], "s", ms=8, color="#ff7f0e", zorder=3)
        ax.annotate("", xy=(tt, y - 0.10), xytext=(uu, y + 0.10),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
        ax.text(uu, y + 0.42, f"$u_{i+1}$={uu:.3f}", ha="center", fontsize=9, color="#1f77b4")
        ax.text(tt, y - 0.48, f"$t_{{d{i+1}}}$={tt:.3f}", ha="center", fontsize=9, color="#ff7f0e")
        if i > 0:
            gap = td[i] - td[i - 1]
            ax.annotate(f"间隔 {gap:.2f} s", xy=((td[i-1] + td[i]) / 2, y + 0.18),
                        xytext=((td[i-1] + td[i]) / 2, y + 0.62), ha="center",
                        fontsize=8, color="#2ca02c")
    ax.set_xlim(-0.5, 8.0)
    ax.set_ylim(0.0, 1.8)
    ax.set_xlabel("时间 (s)")
    ax.set_yticks([])
    ax.set_title(r"结构化编码：排序变量 $u$ 映射为投放时刻 $t_d$（间隔 $\geq 1$ s）", fontsize=10.5)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out)


def fig_heuristic(out: Path) -> None:
    spec = missile_spec("M1")
    tb = 15.0
    p = missile_position(spec.p0, spec.direction, tb)
    e = p + 0.3 * (TARGET_CENTER - p)
    u0 = np.array([17800.0, 0.0])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot([p[0], TARGET_CENTER[0]], [p[1], TARGET_CENTER[1]], color="#1f77b4",
            lw=2, label=f"$t_b={tb:.0f}$ s 时导弹—目标视线")
    ax.plot(*u0, marker="^", ms=11, color="#9467bd", label="FY1 初始位置")
    ax.plot([u0[0], e[0]], [u0[1], e[1]], ls="--", color="#9467bd", lw=1.2,
            label="无人机水平航向")
    ax.plot(*e[:2], marker="*", ms=15, color="#ff7f0e", label=r"期望起爆点 $E^{des}$")
    ax.annotate("", xy=(e[0], e[1]), xytext=(p[0], p[1]),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.3))
    ax.text(0.35 * p[0] + 0.65 * e[0], 0.35 * p[1] + 0.65 * e[1] + 18,
            "$\\lambda=0.3$", color="#2ca02c", fontsize=10)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_title("几何启发式：将起爆点置于导弹视线附近", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    _save(fig, out)


def fig_gpu_kernel(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    boxes = [
        (0.02, 0.25, 0.18, 0.5, "种群 $X$\n$(P\\times40)$"),
        (0.32, 0.25, 0.30, 0.5, "Triton 核函数\n网格 $(P,3)$ 个程序\n每程序：15 弹 $\\times$ 700 时刻"),
        (0.74, 0.25, 0.22, 0.5, "时长矩阵\n$(P\\times3)$\n$\\rightarrow$ 能量 $E$"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                    fc="#eaf2fb", ec="#1f77b4", lw=1.4))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    for x1, x2 in [(0.20, 0.32), (0.62, 0.74)]:
        ax.add_patch(FancyArrowPatch((x1, 0.5), (x2, 0.5), arrowstyle="-|>",
                                     mutation_scale=18, color="#d62728", lw=1.6))
    ax.text(0.26, 0.62, "一次启动", fontsize=9, color="#d62728", ha="center")
    ax.text(0.68, 0.62, "逐行求和", fontsize=9, color="#d62728", ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("GPU 批量目标评估：一个程序处理一个（候选解, 导弹）", fontsize=11)
    _save(fig, out)


def main() -> None:
    outdir = ROOT / "figures"
    fig_interval_refine(outdir / "fig5_interval_refine.png")
    fig_encoding(outdir / "fig5_encoding.png")
    fig_heuristic(outdir / "fig5_heuristic.png")
    fig_gpu_kernel(outdir / "fig5_gpu_kernel.png")
    paper_figs = Path("/home/violet/Workspace/Code/Text_code/数模/2025_cumcm_A/figs")
    paper_figs.mkdir(parents=True, exist_ok=True)
    for name in ("fig5_interval_refine.png", "fig5_encoding.png",
                 "fig5_heuristic.png", "fig5_gpu_kernel.png"):
        (paper_figs / name).write_bytes((outdir / name).read_bytes())
    print("figures written to", outdir)


if __name__ == "__main__":
    main()
