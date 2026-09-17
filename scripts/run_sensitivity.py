#!/usr/bin/env python3
"""Run sensitivity analysis and generate 3-D coupling figures."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sensitivity import plot_sensitivity_3d, sensitivity_table


def main() -> None:
    res = str(ROOT / "results")
    fig = str(ROOT / "figures")
    table = sensitivity_table(res)
    print("Q2 3D:", plot_sensitivity_3d(2, res, fig))
    print("Q5 3D:", plot_sensitivity_3d(5, res, fig))
    for q, row in table["table"].items():
        print(q, row)


if __name__ == "__main__":
    main()
