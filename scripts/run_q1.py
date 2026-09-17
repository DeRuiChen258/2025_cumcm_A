#!/usr/bin/env python3
"""Run Q1 direct calculation."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting import plot_gantt, plot_strategy_3d
from src.problems import solve_q1


def main() -> None:
    results_dir = ROOT / "results"
    fig_dir = ROOT / "figures"
    q1 = solve_q1(str(results_dir))
    print(json.dumps(q1, ensure_ascii=False, indent=2))
    plot_gantt({"M1": q1["intervals"]}, str(fig_dir / "q1_gantt.png"),
               title="Q1 shielding interval of M1")
    strat = q1["strategy"]
    plot_strategy_3d(strat, ("M1",), str(fig_dir / "q1_traj3d.png"),
                     title="Q1 trajectory (fixed FY1 strategy)")


if __name__ == "__main__":
    main()
