#!/usr/bin/env python3
"""Run Q2 single-shell optimization."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.plotting import plot_convergence, plot_gantt, plot_strategy_3d
from src.problems import solve_q2


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    q2 = solve_q2(cfg["budgets"], str(ROOT / "results"))
    print(json.dumps({k: v for k, v in q2.items() if k != "strategy"}, ensure_ascii=False, indent=2))
    plot_gantt({"M1": q2["intervals"]}, str(ROOT / "figures/q2_gantt.png"),
               title="Q2 optimal shielding interval")
    plot_strategy_3d(q2["strategy"], ("M1",), str(ROOT / "figures/q2_traj3d.png"),
                     title="Q2 optimal trajectory")
    plot_convergence(q2["objective_history"], str(ROOT / "figures/q2_convergence.png"),
                     title="Q2 DE convergence")


if __name__ == "__main__":
    main()
