#!/usr/bin/env python3
"""Run Q4 three-UAV optimization and export result2.xlsx."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.export import export_result2
from src.plotting import plot_convergence, plot_gantt, plot_strategy_3d
from src.problems import solve_q4


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    q4 = solve_q4(cfg["budgets"], str(ROOT / "results"))
    print(json.dumps({k: v for k, v in q4.items() if k != "strategy"}, ensure_ascii=False, indent=2))
    plot_gantt({"M1 (union)": q4["intervals"]}, str(ROOT / "figures/q4_gantt.png"),
               title="Q4 union shielding interval (3 UAVs)")
    plot_strategy_3d(q4["strategy"], ("M1",), str(ROOT / "figures/q4_traj3d.png"),
                     title="Q4 three-UAV strategy")
    plot_convergence(q4["objective_history"], str(ROOT / "figures/q4_convergence.png"),
                     title="Q4 DE convergence")
    export_result2(q4, str(ROOT / "results/result2.xlsx"))


if __name__ == "__main__":
    main()
