"""Sensitivity analysis and 3-D coupling plots."""

from __future__ import annotations

import json
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import MISSILE_NAMES
from .objective import evaluate_strategy


def _load_strategy(results_dir: str, key: str) -> Dict:
    with open(os.path.join(results_dir, key), encoding="utf-8") as f:
        return json.load(f)


def sensitivity_table(results_dir: str = "results") -> Dict:
    problems = {
        1: ("q1_best.json", ("M1",)),
        2: ("q2_best.json", ("M1",)),
        3: ("q3_best.json", ("M1",)),
        4: ("q4_best.json", ("M1",)),
        5: ("q5_best.json", tuple(MISSILE_NAMES)),
    }
    variants = [
        ("baseline", {}),
        ("g=10.0", {"g_override": 10.0}),
        ("R=9.0", {"radius_override": 9.0}),
        ("R=11.0", {"radius_override": 11.0}),
        ("min_h=30", {"min_height_override": 30.0}),
        ("cylinder_criterion", {"criterion": "cylinder"}),
    ]
    table: Dict[str, Dict[str, float]] = {}
    for p, (fname, mkeys) in problems.items():
        r = _load_strategy(results_dir, fname)
        strat = r["strategy"]
        row: Dict[str, float] = {}
        for name, kwargs in variants:
            ev = evaluate_strategy(strat, mkeys, refine=True, **kwargs)
            if ev["valid"]:
                row[name] = round(ev["total"], 3)
            else:
                row[name] = None
        table[f"Q{p}"] = row
    out = {"variants": [v[0] for v in variants], "table": table}
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "sensitivity_table.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def plot_sensitivity_3d(problem: int, results_dir: str = "results",
                        fig_dir: str = "figures",
                        g_range=(8.8, 10.8), r_range=(8.0, 12.0), n: int = 11) -> str:
    fname = {2: "q2_best.json", 5: "q5_best.json"}[problem]
    r = _load_strategy(results_dir, fname)
    strat = r["strategy"]
    mkeys = ("M1",) if problem == 2 else tuple(MISSILE_NAMES)
    gs = np.linspace(*g_range, n)
    rs = np.linspace(*r_range, n)
    Z = np.zeros((n, n))
    for i, g in enumerate(gs):
        for j, rad in enumerate(rs):
            ev = evaluate_strategy(strat, mkeys, refine=False,
                                   g_override=float(g), radius_override=float(rad))
            Z[i, j] = ev["total"] if ev["valid"] else 0.0
    Gg, Rr = np.meshgrid(gs, rs, indexing="ij")
    fig = plt.figure(figsize=(7.5, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Gg, Rr, Z, cmap="viridis", edgecolor="none", alpha=0.95)
    ax.set_xlabel("g (m/s$^2$)")
    ax.set_ylabel("Cloud radius R (m)")
    ax.set_zlabel("Shielded duration (s)")
    ax.set_title(f"Sensitivity of Q{problem} strategy", fontsize=11)
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, f"fig_sens3d_q{problem}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
