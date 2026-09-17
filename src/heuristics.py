"""Heuristic feasible seeds for DE initial populations."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import (CLOUD_RADIUS, DROP_TIME_RANGE, G, MISSILE_NAMES, MISSILE_SPEED,
                     TARGET_CENTER, TAU_MIN, UAVS, UAV_NAMES, UAV_SPEED_RANGE,
                     missile_spec, uav_spec)
from .kinematics import missile_position
from .objective import evaluate_strategy


def _single_seed_candidates(uav_idx: int, missile_key: str,
                            t_bursts: Sequence[float],
                            fracs: Sequence[float],
                            speeds: Sequence[float]) -> List[Tuple[float, float, float, float]]:
    """Generate (theta, v, t_d, tau) candidates whose burst point lies on the
    missile sightline at t_burst and is reachable from the UAV."""
    uav_name = UAV_NAMES[uav_idx]
    u0 = UAVS[uav_name].astype(float)
    h = uav_spec(uav_name).height
    tau_max = uav_spec(uav_name).tau_max
    spec = missile_spec(missile_key)
    out: List[Tuple[float, float, float, float]] = []
    for tb in t_bursts:
        p = missile_position(spec.p0, spec.direction, float(tb))
        for frac in fracs:
            e = p + frac * (TARGET_CENTER - p)
            ez = float(e[2])
            if ez < 60.0:
                continue
            tau = float(np.sqrt(max(0.0, 2.0 * (h - ez) / G)))
            if not (TAU_MIN <= tau <= tau_max):
                continue
            delta = e[:2] - u0[:2]
            s = float(np.linalg.norm(delta))
            if s < 1.0:
                continue
            theta = float(np.arctan2(delta[1], delta[0])) % (2.0 * np.pi)
            for v in speeds:
                td = s / v - tau
                if DROP_TIME_RANGE[0] <= td <= DROP_TIME_RANGE[1]:
                    out.append((theta, v, td, tau))
    return out


def single_shell_objective(uav_idx: int, missile_key: str, x4: Sequence[float],
                           refine: bool = False) -> float:
    theta, v, td, tau = (float(a) for a in x4)
    strat = {"uavs": {uav_idx: {"theta": theta, "v": v}},
             "shells": [{"uav": uav_idx, "missile": 0, "t_d": td, "tau": tau,
                         "t_e": td + tau}]}
    ev = evaluate_strategy(strat, (missile_key,), refine=refine)
    return ev["durations"][missile_key] if ev["valid"] else 0.0


def best_single_seed(uav_idx: int, missile_key: str) -> Tuple[float, float, float, float]:
    t_bursts = np.linspace(4.0, 55.0, 18)
    fracs = [0.05, 0.15, 0.3, 0.5, 0.7, 0.9]
    speeds = [70.0, 90.0, 110.0, 130.0, 140.0]
    cands = _single_seed_candidates(uav_idx, missile_key, t_bursts, fracs, speeds)
    scored = [(single_shell_objective(uav_idx, missile_key, c), c) for c in cands]
    scored.sort(key=lambda p: -p[0])
    if not scored:
        # fallback: Q1-style parameters adapted to the UAV
        return (np.pi, 120.0, 1.5, min(3.6, uav_spec(UAV_NAMES[uav_idx]).tau_max))
    return scored[0][1]


def q2_seeds() -> List[np.ndarray]:
    t_bursts = np.linspace(4.0, 55.0, 24)
    fracs = [0.05, 0.15, 0.3, 0.5, 0.7, 0.9]
    speeds = [70.0, 90.0, 110.0, 130.0, 140.0]
    cands = _single_seed_candidates(0, "M1", t_bursts, fracs, speeds)
    scored = [(single_shell_objective(0, "M1", c), c) for c in cands]
    scored.sort(key=lambda p: -p[0])
    return [np.array(c) for _, c in scored[:12]]


def q3_seeds() -> List[np.ndarray]:
    singles = q2_seeds()
    seeds = []
    for i in range(min(len(singles), 8)):
        theta, v = singles[i][0], singles[i][1]
        others = [s for j, s in enumerate(singles) if j != i][:2]
        while len(others) < 2:
            others.append((theta, v, 1.5, 3.6))
        u = np.array([singles[i][2], others[0][2], others[1][2]])
        tau = np.array([singles[i][3], others[0][3], others[1][3]])
        seeds.append(np.concatenate([[theta, v], u, tau]))
    # diverse variants with different theta/v pairs
    for a in range(min(len(singles), 4)):
        for b in range(min(len(singles), 4)):
            if a == b:
                continue
            theta = (singles[a][0] + singles[b][0]) / 2.0
            v = (singles[a][1] + singles[b][1]) / 2.0
            u = np.array([singles[a][2], singles[b][2], singles[a][2] + 5.0])
            tau = np.array([singles[a][3], singles[b][3], singles[a][3]])
            seeds.append(np.concatenate([[theta, v], u, tau]))
    return seeds


def q4_seeds() -> List[np.ndarray]:
    seeds = []
    for k in range(3):
        seeds.append(best_single_seed(k, "M1"))
    base = np.concatenate([np.array(s) for s in seeds])
    out = [base]
    # variants: rotate representative seed for each UAV
    for offset in range(1, 4):
        parts = []
        for k in range(3):
            s = best_single_seed(k, "M1")
            if k == offset % 3:
                s = best_single_seed(k, "M2") if k != 2 else best_single_seed(k, "M3")
            parts.append(np.array(s))
        out.append(np.concatenate(parts))
    return out


def q5_seeds() -> List[np.ndarray]:
    """Build global 40-D seeds from per-(UAV, missile) single-shell heuristics."""
    best = {}
    for k in range(5):
        for m in range(3):
            best[(k, m)] = best_single_seed(k, MISSILE_NAMES[m])

    seeds = []
    for rep_m in range(3):  # choose representative missile for each UAV's theta/v
        x = []
        for k in range(5):
            rep = best[(k, rep_m)]
            theta, v = rep[0], rep[1]
            tds = [best[(k, m)][2] for m in range(3)]
            taus = [best[(k, m)][3] for m in range(3)]
            order = np.argsort(tds)
            tds_sorted = np.array(tds)[order]
            taus_sorted = np.array(taus)[order]
            u = tds_sorted - np.arange(3, dtype=float)
            u = np.clip(u, DROP_TIME_RANGE[0], DROP_TIME_RANGE[1])
            x += [theta, v] + list(u) + list(taus_sorted)
        seeds.append(np.array(x))

    # variant: mix representative missiles across UAVs
    for k in range(5):
        x = []
        for kk in range(5):
            rep_m = (k + kk) % 3
            rep = best[(kk, rep_m)]
            theta, v = rep[0], rep[1]
            tds = [best[(kk, m)][2] for m in range(3)]
            taus = [best[(kk, m)][3] for m in range(3)]
            order = np.argsort(tds)
            u = np.clip(np.array(tds)[order] - np.arange(3, dtype=float),
                        DROP_TIME_RANGE[0], DROP_TIME_RANGE[1])
            x += [theta, v] + list(u) + list(np.array(taus)[order])
        seeds.append(np.array(x))
    return seeds


def make_initial_population(problem: int, objective, bounds: Sequence[Tuple[float, float]],
                            popsize: int, seed: int = 2025,
                            n_random: int = 4000) -> np.ndarray:
    """Return an initial population (popsize, dim) mixing heuristics and random samples."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    if problem == 2:
        seeds = q2_seeds()
    elif problem == 3:
        seeds = q3_seeds()
    elif problem == 4:
        seeds = q4_seeds()
    else:
        seeds = q5_seeds()

    random = rng.uniform(lo, hi, size=(n_random, len(bounds)))
    all_candidates = np.vstack([np.asarray(s, dtype=float) for s in seeds] + [random])
    vals = np.array([objective(x) for x in all_candidates], dtype=float)
    order = np.argsort(vals)
    pop = all_candidates[order[:popsize]].copy()
    if pop.shape[0] < popsize:
        pad = rng.uniform(lo, hi, size=(popsize - pop.shape[0], len(bounds)))
        pop = np.vstack([pop, pad])
    return pop
