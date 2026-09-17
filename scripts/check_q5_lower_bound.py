#!/usr/bin/env python3
"""Cross-check the Q5 solution against a better feasible strategy.

问题 4 得到的 FY1--FY3 各一弹策略在问题 5 的可行域内同样可行。把问题 5
解中 FY1--FY3 的弹替换为问题 4 的解、保留 FY4/FY5 的弹，可得到一个目标值
更高的可行方案；这说明 ``results/q5_best.json`` 中的解不是全局最优。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.objective import MISSILE_NAMES, evaluate_strategy


def uav_config(strategy: dict, index: int) -> dict:
    """Return the (theta, v) config of one UAV, tolerating str/int keys."""
    uavs = strategy["uavs"]
    return uavs.get(index, uavs.get(str(index)))


def shells_of(strategy: dict, index: int) -> list:
    return [s for s in strategy["shells"] if int(s["uav"]) == index]


def main() -> None:
    q4 = json.loads((ROOT / "results/q4_best.json").read_text(encoding="utf-8"))
    q5 = json.loads((ROOT / "results/q5_best.json").read_text(encoding="utf-8"))

    uavs: dict = {}
    shells: list = []
    for k in range(3):  # FY1--FY3 沿用问题 4 的 M1 单弹策略
        cfg = uav_config(q4["strategy"], k)
        uavs[k] = {"theta": cfg["theta"], "v": cfg["v"]}
        sh = shells_of(q4["strategy"], k)[0]
        shells.append({"uav": k, "missile": 0, "t_d": sh["t_d"], "tau": sh["tau"],
                       "t_e": sh["t_d"] + sh["tau"]})
    for k in (3, 4):  # FY4/FY5 沿用问题 5 解中的弹
        cfg = uav_config(q5["strategy"], k)
        uavs[k] = {"theta": cfg["theta"], "v": cfg["v"]}
        for sh in shells_of(q5["strategy"], k):
            shells.append({"uav": k, "missile": int(sh["missile"]),
                           "t_d": sh["t_d"], "tau": sh["tau"], "t_e": sh["t_e"]})

    ev = evaluate_strategy({"uavs": uavs, "shells": shells},
                           tuple(MISSILE_NAMES), refine=True)
    print(f"组合方案（{len(shells)} 弹）: 可行={ev['valid']} "
          f"总遮蔽={ev['total']:.3f} s "
          f"(M1 {ev['durations']['M1']:.3f} / "
          f"M2 {ev['durations']['M2']:.3f} / "
          f"M3 {ev['durations']['M3']:.3f})")
    print(f"results/q5_best.json 记录的解: {q5['total_duration_s']:.3f} s")


if __name__ == "__main__":
    main()
