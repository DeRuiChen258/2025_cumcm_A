"""Figure generation for results and sensitivity analysis."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLOUD_LIFETIME, MISSILES, MISSILE_NAMES, MISSILE_SPEED, TARGET_CENTER, UAVS
from .kinematics import burst_point, cloud_center, missile_position, shell_position, uav_position
from .objective import decode_q2, decode_q3, decode_q4, decode_q5


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gantt(intervals: Dict[str, List[Tuple[float, float]]], path: str,
               title: str = "", xlim: Tuple[float, float] | None = None) -> str:
    """Horizontal Gantt-style interval chart."""
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.55 * max(len(intervals), 1)))
    labels = list(intervals.keys())
    for i, lab in enumerate(labels):
        y = len(labels) - 1 - i
        for a, b in intervals[lab]:
            ax.barh(y, b - a, left=a, height=0.55, color="#1f77b4",
                    edgecolor="black", linewidth=0.4)
        ax.text(0.99, y, lab, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Time t (s)")
    if xlim:
        ax.set_xlim(*xlim)
    if title:
        ax.set_title(title, fontsize=11)
    _save(fig, path)
    return path


def plot_strategy_3d(strategy: Dict, missile_keys: Sequence[str] = ("M1",),
                     path: str = "", title: str = "") -> str:
    """3D trajectory plot: missiles, UAV paths, shell arcs and cloud tracks."""
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"M1": "#d62728", "M2": "#ff7f0e", "M3": "#2ca02c"}
    for m in missile_keys:
        p0 = MISSILES[m]
        dist = np.linalg.norm(p0)
        u = -p0 / dist
        t_imp = dist / MISSILE_SPEED
        tt = np.linspace(0, t_imp, 80)
        pp = p0[None, :] + MISSILE_SPEED * u[None, :] * tt[:, None]
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], color=colors.get(m, "gray"),
                lw=1.6, label=f"Missile {m}")

    uav_colors = ["#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for k, uav_cfg in strategy["uavs"].items():
        name = list(UAVS.keys())[int(k)]
        p0 = UAVS[name]
        theta = uav_cfg["theta"]
        v = uav_cfg["v"]
        max_td = max((s["t_d"] for s in strategy["shells"] if s["uav"] == k), default=0.0)
        tt = np.linspace(0, min(max_td + 2.0, 65.0), 40)
        uu = p0[None, :] + v * np.array([np.cos(theta), np.sin(theta), 0.0]) * tt[:, None]
        ax.plot(uu[:, 0], uu[:, 1], uu[:, 2], color=uav_colors[int(k) % len(uav_colors)],
                lw=1.2, ls="--", label=f"UAV {name}")

    for i, shell in enumerate(strategy["shells"]):
        k = int(shell["uav"])
        uav_name = list(UAVS.keys())[k]
        uav_cfg = strategy["uavs"][k]
        theta = uav_cfg["theta"]
        v = uav_cfg["v"]
        td = shell["t_d"]
        tau = shell["tau"]
        te = td + tau
        drop = uav_position(UAVS[uav_name], theta, v, td)
        burst = burst_point(drop, theta, v, tau)
        tr = np.linspace(0, tau, 20)
        sp = np.stack([shell_position(drop, theta, v, trr) for trr in tr])
        ax.plot(sp[:, 0], sp[:, 1], sp[:, 2], color="gray", lw=0.9, alpha=0.8)
        tc = np.linspace(0, min(CLOUD_LIFETIME, 25.0), 25)
        cc = np.stack([cloud_center(burst, tcr) for tcr in tc])
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], color="#3b7dd8", lw=1.4,
                alpha=0.75, label="Cloud track" if i == 0 else None)
        ax.scatter(*burst, s=18, color="#3b7dd8", alpha=0.9)

    ax.scatter(*TARGET_CENTER, s=60, marker="*", color="k", label="True target")
    ax.scatter([0], [0], [0], s=45, marker="x", color="r", label="Fake target")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    _save(fig, path)
    return path


def plot_convergence(history: Sequence[float], path: str, title: str = "") -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(np.arange(1, len(history) + 1), history, lw=1.5, color="#1f77b4")
    ax.set_xlabel("DE iteration")
    ax.set_ylabel("Best shielded duration (s)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    _save(fig, path)
    return path


def strategy_for_problem(problem: int, x: Sequence[float]) -> Dict:
    if problem == 2:
        return decode_q2(x)
    if problem == 3:
        return decode_q3(x)
    if problem == 4:
        return decode_q4(x)
    if problem == 5:
        return decode_q5(x)
    raise ValueError(problem)
