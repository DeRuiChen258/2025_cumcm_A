"""Export result1-3.xlsx with the default column specification."""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd


COMMON_COLS = ["弹编号", "无人机", "速度 v (m/s)", "航向角 θ (rad)",
               "投放时刻 t_d (s)", "起爆时刻 t_e (s)",
               "投放点 x", "投放点 y", "投放点 z",
               "起爆点 x", "起爆点 y", "起爆点 z", "遮蔽时长 (s)"]

RESULT3_COLS = ["弹编号", "无人机", "目标导弹", "速度 v (m/s)", "航向角 θ (rad)",
                "投放时刻 t_d (s)", "起爆时刻 t_e (s)",
                "投放点 x", "投放点 y", "投放点 z",
                "起爆点 x", "起爆点 y", "起爆点 z",
                "贡献遮蔽区间 (s)", "遮蔽时长 (s)"]


def _frame(rows: List[Dict], cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=cols)


def export_result1(q3_result: Dict, path: str) -> str:
    rows = [{k: v for k, v in r.items() if k != "目标导弹" and k != "贡献遮蔽区间 (s)"}
            for r in q3_result["shell_rows"]]
    df = _frame(rows, COMMON_COLS)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_excel(path, index=False)
    return path


def export_result2(q4_result: Dict, path: str) -> str:
    rows = [{k: v for k, v in r.items() if k != "目标导弹" and k != "贡献遮蔽区间 (s)"}
            for r in q4_result["shell_rows"]]
    df = _frame(rows, COMMON_COLS)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_excel(path, index=False)
    return path


def export_result3(q5_result: Dict, path: str) -> str:
    df = _frame(q5_result["shell_rows"], RESULT3_COLS)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_excel(path, index=False)
    return path
