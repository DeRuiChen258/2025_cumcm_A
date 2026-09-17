#!/usr/bin/env python3
"""Run Q1-Q5 end-to-end and export all artifacts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.problems import solve_q1, solve_q2, solve_q3, solve_q4, solve_q5


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    budgets = cfg["budgets"]
    results = str(ROOT / "results")
    solve_q1(results)
    solve_q2(budgets, results)
    solve_q3(budgets, results)
    solve_q4(budgets, results)
    solve_q5(budgets, results)
    print("All problems solved. See results/*.json and results/result*.xlsx")


if __name__ == "__main__":
    main()
