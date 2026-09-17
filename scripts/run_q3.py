#!/usr/bin/env python3
"""Run Q3 three-shell optimization and export result1.xlsx."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.export import export_result1
from src.plotting import plot_convergence, plot_gantt, plot_strategy_3d
from src.problems import solve_q3


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    q3 = solve_q3(cfg["budgets"], str(ROOT / "results"))
    print(json.dumps({k: v for k, v in q3.items() if k != "strategy"}, ensure_ascii=False, indent=2))
    plot_gantt({"M1 (union)": q3["intervals"]}, str(ROOT / "figures/q3_gantt.png"),
               title="Q3 union shielding interval (3 shells)")
    plot_strategy_3d(q3["strategy"], ("M1",), str(ROOT / "figures/q3_traj3d.png"),
                     title="Q3 three-shell strategy")
    plot_convergence(q3["objective_history"], str(ROOT / "figures/q3_convergence.png"),
                     title="Q3 DE convergence")
    export_result1(q3, str(ROOT / "results/result1.xlsx"))


if __name__ == "__main__":
    main()
