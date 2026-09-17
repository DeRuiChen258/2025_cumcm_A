"""Q1-Q5 solution entry points."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from .config import (CLOUD_RADIUS, MISSILES, MISSILE_NAMES, TARGET_CENTER,
                     UAV_NAMES, UAVS, missile_spec, uav_spec)
from .geometry import total_length
from .kinematics import burst_point, uav_position
from .objective import (bounds_q2, bounds_q3, bounds_q4, bounds_q5, decode_q2,
                        decode_q3, decode_q4, decode_q5, evaluate_strategy,
                        q2_objective, q3_objective, q4_objective, q5_objective)
from .optimize import local_refine, multistart, run_de
from .shielding import shielded_intervals
from .heuristics import make_initial_population


def _shell_rows(strategy: Dict, ev: Dict, missile_keys: List[str]) -> List[Dict]:
    rows = []
    for i, shell in enumerate(strategy["shells"]):
        uav_idx = int(shell["uav"])
        uav_name = UAV_NAMES[uav_idx]
        cfg = strategy["uavs"][uav_idx] if isinstance(
            strategy["uavs"].get(uav_idx), dict) else strategy["uavs"][str(uav_idx)]
        theta = float(cfg["theta"])
        v = float(cfg["v"])
        td = float(shell["t_d"])
        tau = float(shell["tau"])
        te = td + tau
        drop = uav_position(UAVS[uav_name], theta, v, td)
        burst = burst_point(drop, theta, v, tau)
        target = shell["missile"]
        mkey = missile_keys[target]
        iv = ev["per_shell"][mkey][i]
        rows.append({
            "弹编号": i + 1,
            "无人机": uav_name,
            "目标导弹": mkey,
            "速度 v (m/s)": round(v, 3),
            "航向角 θ (rad)": round(theta, 5),
            "投放时刻 t_d (s)": round(td, 3),
            "起爆时刻 t_e (s)": round(te, 3),
            "投放点 x": round(float(drop[0]), 2),
            "投放点 y": round(float(drop[1]), 2),
            "投放点 z": round(float(drop[2]), 2),
            "起爆点 x": round(float(burst[0]), 2),
            "起爆点 y": round(float(burst[1]), 2),
            "起爆点 z": round(float(burst[2]), 2),
            "贡献遮蔽区间 (s)": "[" + ", ".join(f"[{a:.3f}, {b:.3f}]" for a, b in iv) + "]",
            "遮蔽时长 (s)": round(total_length(iv), 3),
        })
    return rows


def solve_q1(results_dir: str = "results") -> Dict:
    """Q1: fixed FY1 strategy, direct calculation."""
    theta = np.pi
    v = 120.0
    td = 1.5
    tau = 3.6
    te = td + tau
    drop = uav_position(UAVS["FY1"], theta, v, td)
    burst = burst_point(drop, theta, v, tau)
    strat = {"uavs": {0: {"theta": theta, "v": v}},
             "shells": [{"uav": 0, "missile": 0, "t_d": td, "tau": tau, "t_e": te}]}
    ev = evaluate_strategy(strat, ("M1",), refine=True)
    iv = ev["unions_by_missile"]["M1"]
    result = {
        "problem": 1,
        "strategy": strat,
        "drop_point": drop.tolist(),
        "burst_point": burst.tolist(),
        "intervals": [[a, b] for a, b in iv],
        "duration_s": round(ev["durations"]["M1"], 6),
        "burst_height_m": round(float(burst[2]), 3),
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "q1_best.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def solve_q2(budgets: Dict, results_dir: str = "results") -> Dict:
    bounds = bounds_q2()
    pop_count = budgets["q2"]["popsize"] * max(1, len(bounds) - 1)
    init = make_initial_population(2, lambda x: q2_objective(x, refine=False),
                                   bounds, pop_count)
    x, f, hist = run_de(lambda x: q2_objective(x, refine=False), bounds_q2(),
                        popsize=budgets["q2"]["popsize"],
                        maxiter=budgets["q2"]["maxiter"], init=init)
    x, f = local_refine(lambda x: q2_objective(x, refine=True), x, bounds_q2())
    # extra multi-start guard on the refined basin
    x2, f2 = multistart(lambda x: q2_objective(x, refine=True), bounds_q2(), n_starts=24)
    if f2 < f:
        x, f = x2, f2
    strat = decode_q2(x)
    ev = evaluate_strategy(strat, ("M1",), refine=True)
    result = {
        "problem": 2,
        "x": [round(float(v_), 6) for v_ in x],
        "strategy": strat,
        "duration_s": round(ev["durations"]["M1"], 6),
        "intervals": [[a, b] for a, b in ev["unions_by_missile"]["M1"]],
        "objective_history": [round(float(v_), 6) for v_ in hist],
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "q2_best.json"), "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    return result


def solve_q3(budgets: Dict, results_dir: str = "results") -> Dict:
    bounds = bounds_q3()
    pop_count = budgets["q3"]["popsize"] * max(1, len(bounds) - 1)
    init = make_initial_population(3, lambda x: q3_objective(x, refine=False),
                                   bounds, pop_count)
    x, f, hist = run_de(lambda x: q3_objective(x, refine=False), bounds_q3(),
                        popsize=budgets["q3"]["popsize"],
                        maxiter=budgets["q3"]["maxiter"], init=init)
    x, f = local_refine(lambda x: q3_objective(x, refine=True), x, bounds_q3())
    strat = decode_q3(x)
    ev = evaluate_strategy(strat, ("M1",), refine=True)
    result = {
        "problem": 3,
        "x": [round(float(v_), 6) for v_ in x],
        "strategy": strat,
        "duration_s": round(ev["durations"]["M1"], 6),
        "intervals": [[a, b] for a, b in ev["unions_by_missile"]["M1"]],
        "shell_rows": _shell_rows(strat, ev, ["M1"]),
        "objective_history": [round(float(v_), 6) for v_ in hist],
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "q3_best.json"), "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    return result


def solve_q4(budgets: Dict, results_dir: str = "results") -> Dict:
    bounds = bounds_q4()
    pop_count = budgets["q4"]["popsize"] * max(1, len(bounds) - 1)
    init = make_initial_population(4, lambda x: q4_objective(x, refine=False),
                                   bounds, pop_count)
    x, f, hist = run_de(lambda x: q4_objective(x, refine=False), bounds_q4(),
                        popsize=budgets["q4"]["popsize"],
                        maxiter=budgets["q4"]["maxiter"], init=init)
    x, f = local_refine(lambda x: q4_objective(x, refine=True), x, bounds_q4())
    strat = decode_q4(x)
    ev = evaluate_strategy(strat, ("M1",), refine=True)
    result = {
        "problem": 4,
        "x": [round(float(v_), 6) for v_ in x],
        "strategy": strat,
        "duration_s": round(ev["durations"]["M1"], 6),
        "intervals": [[a, b] for a, b in ev["unions_by_missile"]["M1"]],
        "shell_rows": _shell_rows(strat, ev, ["M1"]),
        "objective_history": [round(float(v_), 6) for v_ in hist],
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "q4_best.json"), "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    return result


def solve_q5(budgets: Dict, results_dir: str = "results") -> Dict:
    bounds = bounds_q5()
    pop_count = budgets["q5"]["popsize"] * max(1, len(bounds) - 1)
    init = make_initial_population(5, lambda x: q5_objective(x, refine=False),
                                   bounds, pop_count, n_random=3000)
    prev_x, prev_total = _load_previous_q5(results_dir)
    if prev_x is not None:
        init[0] = np.asarray(prev_x, dtype=float)
    if budgets["q5"].get("gpu", False):
        try:
            from .optimize_gpu import run_gpu_de_q5
            x, f, hist = run_gpu_de_q5(bounds, popsize=budgets["q5"]["popsize"],
                                       maxiter=budgets["q5"]["maxiter"], seed=2025, init=init)
        except Exception:
            x, f, hist = run_de(lambda x: q5_objective(x, refine=False), bounds_q5(),
                                popsize=budgets["q5"]["popsize"],
                                maxiter=budgets["q5"]["maxiter"], init=init)
    else:
        x, f, hist = run_de(lambda x: q5_objective(x, refine=False), bounds_q5(),
                            popsize=budgets["q5"]["popsize"],
                            maxiter=budgets["q5"]["maxiter"], init=init)
    x, f = local_refine(lambda x: q5_objective(x, refine=True), x, bounds)
    if prev_x is not None:
        xp, fp = local_refine(lambda x: q5_objective(x, refine=True),
                              np.asarray(prev_x, dtype=float), bounds)
        if fp < f:
            x, f = xp, fp
    strat = decode_q5(x)
    ev = evaluate_strategy(strat, tuple(MISSILE_NAMES), refine=True)
    result = {
        "problem": 5,
        "x": [round(float(v_), 6) for v_ in x],
        "strategy": strat,
        "durations_s": {m: round(ev["durations"][m], 6) for m in MISSILE_NAMES},
        "total_duration_s": round(ev["total"], 6),
        "simultaneous_duration_s": round(ev["simultaneous_duration"], 6),
        "worst_missile_duration_s": round(min(ev["durations"].values()), 6),
        "unions": {m: [[a, b] for a, b in ev["unions_by_missile"][m]]
                   for m in MISSILE_NAMES},
        "shell_rows": _shell_rows(strat, ev, MISSILE_NAMES),
        "objective_history": [round(float(v_), 6) for v_ in hist],
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "q5_best.json"), "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    return result


def _load_previous_q5(results_dir: str):
    """Load a previous Q5 solution (if any) so reruns never regress."""
    path = os.path.join(results_dir, "q5_best.json")
    if not os.path.exists(path):
        return None, -float("inf")
    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        x = np.asarray(r.get("x"), dtype=float)
        total = float(r.get("total_duration_s", -1.0))
        if x.size == 40 and total > 0:
            return x, total
    except Exception:
        pass
    return None, -float("inf")
