#!/usr/bin/env python3
"""Run Q5 full 15-shell optimization and export result3.xlsx."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.export import export_result3
from src.plotting import plot_convergence, plot_gantt, plot_strategy_3d
from src.problems import solve_q5


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    q5 = solve_q5(cfg["budgets"], str(ROOT / "results"))
    print(json.dumps({k: v for k, v in q5.items() if k != "strategy"}, ensure_ascii=False, indent=2))
    plot_gantt(q5["unions"], str(ROOT / "figures/q5_gantt.png"),
               title="Q5 shielding intervals of M1-M3", xlim=(0.0, 70.0))
    plot_strategy_3d(q5["strategy"], ("M1", "M2", "M3"),
                     str(ROOT / "figures/q5_traj3d.png"), title="Q5 15-shell strategy")
    plot_convergence(q5["objective_history"], str(ROOT / "figures/q5_convergence.png"),
                     title="Q5 DE convergence")
    export_result3(q5, str(ROOT / "results/result3.xlsx"))


if __name__ == "__main__":
    main()
