"""Objective functions and decision-variable encodings for Q2-Q5."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import (CLOUD_RADIUS, CYLINDER_HEIGHT, CYLINDER_RADIUS,
                     DROP_TIME_RANGE, G, MIN_BURST_HEIGHT, MISSILE_NAMES,
                     TARGET_CENTER, TAU_MIN,
                     UAV_NAMES, UAV_SPEED_RANGE, uav_spec, missile_spec)
from .geometry import intersect_many, total_length, union_intervals
from .kinematics import burst_point, uav_position
from .shielding import shielded_intervals


def decode_q2(x: Sequence[float]) -> Dict:
    theta, v, u, tau = (float(v_) for v_ in x)
    return {
        "uavs": {0: {"theta": theta, "v": v}},
        "shells": [{"uav": 0, "missile": 0, "t_d": u, "tau": tau, "t_e": u + tau}],
    }


def decode_q3(x: Sequence[float]) -> Dict:
    theta, v = float(x[0]), float(x[1])
    u = np.asarray(x[2:5], dtype=float)
    tau = np.asarray(x[5:8], dtype=float)
    order = np.argsort(u)
    us = u[order]
    ts = tau[order]
    t_d = us + np.arange(3, dtype=float)
    shells = [{"uav": 0, "missile": 0, "t_d": float(t_d[i]), "tau": float(ts[i]),
               "t_e": float(t_d[i] + ts[i])} for i in range(3)]
    return {"uavs": {0: {"theta": theta, "v": v}}, "shells": shells}


def decode_q4(x: Sequence[float]) -> Dict:
    uavs: Dict[int, Dict[str, float]] = {}
    shells = []
    for k in range(3):
        theta, v, u, tau = (float(v_) for v_ in x[4 * k: 4 * k + 4])
        uavs[k] = {"theta": theta, "v": v}
        shells.append({"uav": k, "missile": 0, "t_d": u, "tau": tau, "t_e": u + tau})
    return {"uavs": uavs, "shells": shells}


def decode_q5(x: Sequence[float]) -> Dict:
    uavs: Dict[int, Dict[str, float]] = {}
    shells = []
    for k in range(5):
        theta, v = float(x[8 * k]), float(x[8 * k + 1])
        uavs[k] = {"theta": theta, "v": v}
        u = np.asarray(x[8 * k + 2: 8 * k + 5], dtype=float)
        tau = np.asarray(x[8 * k + 5: 8 * k + 8], dtype=float)
        order = np.argsort(u)
        us = u[order]
        ts = tau[order]
        t_d = us + np.arange(3, dtype=float)
        for i in range(3):
            shells.append({"uav": k, "missile": i, "t_d": float(t_d[i]),
                           "tau": float(ts[i]), "t_e": float(t_d[i] + ts[i])})
    return {"uavs": uavs, "shells": shells}


def bounds_q2() -> List[Tuple[float, float]]:
    tau_hi = uav_spec("FY1").tau_max
    return [(0.0, 2.0 * np.pi), UAV_SPEED_RANGE, DROP_TIME_RANGE, (TAU_MIN, tau_hi)]


def bounds_q3() -> List[Tuple[float, float]]:
    tau_hi = uav_spec("FY1").tau_max
    return ([(0.0, 2.0 * np.pi), UAV_SPEED_RANGE]
            + [DROP_TIME_RANGE] * 3 + [(TAU_MIN, tau_hi)] * 3)


def bounds_q4() -> List[Tuple[float, float]]:
    out = []
    for name in UAV_NAMES[:3]:
        tau_hi = uav_spec(name).tau_max
        out += [(0.0, 2.0 * np.pi), UAV_SPEED_RANGE, DROP_TIME_RANGE, (TAU_MIN, tau_hi)]
    return out


def bounds_q5() -> List[Tuple[float, float]]:
    out = []
    for name in UAV_NAMES:
        tau_hi = uav_spec(name).tau_max
        out += ([(0.0, 2.0 * np.pi), UAV_SPEED_RANGE] +
                [DROP_TIME_RANGE] * 3 + [(TAU_MIN, tau_hi)] * 3)
    return out


def evaluate_strategy(strategy: Dict, missile_keys: Sequence[str] = ("M1",),
                      refine: bool = False, g_override: float = G,
                      radius_override: float = CLOUD_RADIUS,
                      min_height_override: float = MIN_BURST_HEIGHT,
                      criterion: str = "point") -> Dict:
    """Compute shielding intervals/durations for a full strategy."""
    from .config import UAVS
    from .shielding import shielded_intervals_cylinder
    uavs = {int(k): v for k, v in strategy["uavs"].items()}
    shell_clouds = []
    for shell in strategy["shells"]:
        uav_idx = int(shell["uav"])
        uav_cfg = uavs[uav_idx]
        theta = float(uav_cfg["theta"])
        v = float(uav_cfg["v"])
        td = float(shell["t_d"])
        tau = float(shell["tau"])
        p0 = UAVS[UAV_NAMES[uav_idx]]
        drop = uav_position(p0, theta, v, td)
        burst = burst_point(drop, theta, v, tau, g_override=g_override)
        shell_clouds.append({"burst": burst, "t_e": td + tau})

    intervals_by_missile: Dict[str, List[Tuple[float, float]]] = {m: [] for m in missile_keys}
    per_shell: Dict[str, Dict[int, List[Tuple[float, float]]]] = {
        m: {i: [] for i in range(len(shell_clouds))} for m in missile_keys}
    valid = True
    for i, sc in enumerate(shell_clouds):
        if sc["burst"][2] < min_height_override - 1e-6:
            valid = False
        for m in missile_keys:
            spec = missile_spec(m)
            if criterion == "cylinder":
                iv = shielded_intervals_cylinder(
                    spec.p0, spec.direction, spec.impact_time,
                    sc["burst"], sc["t_e"],
                    cylinder_center=TARGET_CENTER, radius=CYLINDER_RADIUS,
                    height=CYLINDER_HEIGHT, R=radius_override, refine=refine)
            else:
                iv = shielded_intervals(spec.p0, spec.direction, spec.impact_time,
                                        sc["burst"], sc["t_e"], target=TARGET_CENTER,
                                        R=radius_override, refine=refine)
            intervals_by_missile[m].extend(iv)
            per_shell[m][i] = iv

    unions = {m: union_intervals(intervals_by_missile[m]) for m in missile_keys}
    durations = {m: total_length(unions[m]) for m in missile_keys}
    total = float(sum(durations.values()))
    simultaneous = intersect_many([unions[m] for m in missile_keys])
    return {
        "valid": valid,
        "intervals_by_missile": intervals_by_missile,
        "unions_by_missile": unions,
        "durations": durations,
        "total": total,
        "simultaneous": simultaneous,
        "simultaneous_duration": total_length(simultaneous),
        "per_shell": per_shell,
    }


def q2_objective(x: Sequence[float], refine: bool = False) -> float:
    strat = decode_q2(x)
    ev = evaluate_strategy(strat, ("M1",), refine=refine)
    return -ev["total"] if ev["valid"] else 1e6


def q3_objective(x: Sequence[float], refine: bool = False) -> float:
    strat = decode_q3(x)
    ev = evaluate_strategy(strat, ("M1",), refine=refine)
    return -ev["total"] if ev["valid"] else 1e6


def q4_objective(x: Sequence[float], refine: bool = False) -> float:
    strat = decode_q4(x)
    ev = evaluate_strategy(strat, ("M1",), refine=refine)
    return -ev["total"] if ev["valid"] else 1e6


def q5_objective(x: Sequence[float], refine: bool = False) -> float:
    strat = decode_q5(x)
    ev = evaluate_strategy(strat, tuple(MISSILE_NAMES), refine=refine)
    return -ev["total"] if ev["valid"] else 1e6


DECODERS = {2: decode_q2, 3: decode_q3, 4: decode_q4, 5: decode_q5}
OBJECTIVES = {2: q2_objective, 3: q3_objective, 4: q4_objective, 5: q5_objective}
BOUNDS = {2: bounds_q2, 3: bounds_q3, 4: bounds_q4, 5: bounds_q5}
