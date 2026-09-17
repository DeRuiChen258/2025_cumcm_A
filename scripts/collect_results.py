#!/usr/bin/env python3
"""Collect submission-format result files into a single folder."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src = ROOT / "results"
    dst = ROOT / "提交结果"
    dst.mkdir(exist_ok=True)
    files = ["result1.xlsx", "result2.xlsx", "result3.xlsx"]
    for name in files:
        shutil.copy2(src / name, dst / name)
    readme = dst / "说明.md"
    readme.write_text(
        "# 提交结果（符合题干要求格式）\n\n"
        "| 文件 | 对应问题 | 内容 |\n"
        "|---|---|---|\n"
        "| result1.xlsx | 问题 3 | FY1 投放 3 枚烟幕干扰弹对 M1 干扰的投放策略 |\n"
        "| result2.xlsx | 问题 4 | FY1、FY2、FY3 各投放 1 枚弹对 M1 干扰的投放策略 |\n"
        "| result3.xlsx | 问题 5 | 5 架无人机共 15 枚弹对 M1、M2、M3 干扰的投放策略 |\n\n"
        "## 列格式说明\n\n"
        "### result1.xlsx / result2.xlsx\n\n"
        "弹编号、无人机、速度 v (m/s)、航向角 θ (rad)、投放时刻 t_d (s)、起爆时刻 t_e (s)、"
        "投放点 x/y/z、起爆点 x/y/z、单弹遮蔽时长 (s)。\n\n"
        "### result3.xlsx\n\n"
        "弹编号、无人机、目标导弹、速度 v (m/s)、航向角 θ (rad)、投放时刻 t_d (s)、起爆时刻 t_e (s)、"
        "投放点 x/y/z、起爆点 x/y/z、贡献遮蔽区间 (s)、遮蔽时长 (s)。\n\n"
        "数值与 `results/*.json` 逐位一致，可复现命令见项目 README。\n",
        encoding="utf-8",
    )
    print(f"collected {len(files)} files into {dst}")


if __name__ == "__main__":
    main()
