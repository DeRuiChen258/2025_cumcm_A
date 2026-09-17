"""Shielding interval computation (coarse detection + brentq refinement)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from .config import (CLOUD_LIFETIME, CLOUD_RADIUS, MISSILE_SPEED, SAMPLE_DT,
                     SINK_SPEED, TARGET_CENTER)
from .geometry import dist_to_segment


def _distance_function(p0: np.ndarray, direction: np.ndarray, burst: np.ndarray,
                       t_e: float, target: np.ndarray, R: float):
    def fun(t: float) -> float:
        p = p0 + MISSILE_SPEED * t * direction
        c = burst + SINK_SPEED * (t - t_e) * np.array([0.0, 0.0, -1.0])
        return float(dist_to_segment(c, p, target)) - R
    return fun


def shielded_intervals(p0: np.ndarray, direction: np.ndarray, impact_time: float,
                       burst: np.ndarray, t_e: float, target: np.ndarray = TARGET_CENTER,
                       R: float = CLOUD_RADIUS, dt: float = SAMPLE_DT,
                       refine: bool = True) -> List[Tuple[float, float]]:
    """Return shielding time intervals of one cloud against one missile."""
    t0 = max(0.0, float(t_e))
    t1 = min(float(impact_time), float(t_e) + CLOUD_LIFETIME)
    if t1 - t0 <= 1e-9:
        return []
    n = max(2, int(round((t1 - t0) / dt)) + 1)
    grid = np.linspace(t0, t1, n)
    p = p0[None, :] + MISSILE_SPEED * direction[None, :] * grid[:, None]
    c = burst[None, :] + SINK_SPEED * (grid - t_e)[:, None] * np.array([0.0, 0.0, -1.0])
    d = dist_to_segment(c, p, target)
    active = d <= R
    if not active.any():
        return []

    idx = np.flatnonzero(active)
    runs: List[Tuple[int, int]] = []
    s = int(idx[0])
    prev = s
    for i in idx[1:]:
        i = int(i)
        if i > prev + 1:
            runs.append((s, prev))
            s = i
        prev = i
    runs.append((s, prev))

    fun = _distance_function(p0, direction, burst, t_e, target, R)
    out: List[Tuple[float, float]] = []
    for si, ei in runs:
        a = float(grid[si])
        b = float(grid[ei])
        if refine:
            if si > 0:
                fa = d[si - 1] - R
                fb = d[si] - R
                if fa * fb <= 0.0:
                    a = float(brentq(fun, float(grid[si - 1]), float(grid[si]),
                                     xtol=1e-9, rtol=1e-10))
            if ei < n - 1:
                fa = d[ei] - R
                fb = d[ei + 1] - R
                if fa * fb <= 0.0:
                    b = float(brentq(fun, float(grid[ei]), float(grid[ei + 1]),
                                     xtol=1e-9, rtol=1e-10))
        out.append((a, b))
    return out


def shielded_intervals_cylinder(p0: np.ndarray, direction: np.ndarray,
                                impact_time: float, burst: np.ndarray, t_e: float,
                                cylinder_center: np.ndarray, radius: float,
                                height: float, n_az: int = 16, n_z: int = 3,
                                R: float = CLOUD_RADIUS, dt: float = SAMPLE_DT,
                                refine: bool = True) -> List[Tuple[float, float]]:
    """Strict cylinder criterion: every sightline to sampled cylinder points is blocked."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)
    zs = np.linspace(0.0, height, n_z)
    qs = []
    for z in zs:
        for ang in angles:
            qs.append(cylinder_center + radius * np.array([np.cos(ang), np.sin(ang), 0.0])
                      + np.array([0.0, 0.0, z]))
    qs = np.asarray(qs)  # (K,3)

    t0 = max(0.0, float(t_e))
    t1 = min(float(impact_time), float(t_e) + CLOUD_LIFETIME)
    if t1 - t0 <= 1e-9:
        return []
    n = max(2, int(round((t1 - t0) / dt)) + 1)
    grid = np.linspace(t0, t1, n)
    p = p0[None, :] + MISSILE_SPEED * direction[None, :] * grid[:, None]
    c = burst[None, :] + SINK_SPEED * (grid - t_e)[:, None] * np.array([0.0, 0.0, -1.0])
    # d[t, k]: distance from cloud at time t to segment [P(t), Q_k]
    pc = np.repeat(p[:, None, :], qs.shape[0], axis=1)          # (T,K,3)
    cc = np.repeat(c[:, None, :], qs.shape[0], axis=1)
    qq = np.broadcast_to(qs[None, :, :], pc.shape)
    d = dist_to_segment(cc, pc, qq)                              # (T,K)
    dmax = d.max(axis=1)
    active = dmax <= R
    if not active.any():
        return []
    idx = np.flatnonzero(active)
    runs: List[Tuple[int, int]] = []
    s = int(idx[0])
    prev = s
    for i in idx[1:]:
        i = int(i)
        if i > prev + 1:
            runs.append((s, prev))
            s = i
        prev = i
    runs.append((s, prev))

    def fun(t: float) -> float:
        pt = p0 + MISSILE_SPEED * t * direction
        ct = burst + SINK_SPEED * (t - t_e) * np.array([0.0, 0.0, -1.0])
        dd = dist_to_segment(ct, pt, qs)
        return float(dd.max()) - R

    out: List[Tuple[float, float]] = []
    for si, ei in runs:
        a = float(grid[si])
        b = float(grid[ei])
        if refine:
            if si > 0:
                fa = dmax[si - 1] - R
                fb = dmax[si] - R
                if fa * fb <= 0.0:
                    a = float(brentq(fun, float(grid[si - 1]), float(grid[si]),
                                     xtol=1e-9, rtol=1e-10))
            if ei < n - 1:
                fa = dmax[ei] - R
                fb = dmax[ei + 1] - R
                if fa * fb <= 0.0:
                    b = float(brentq(fun, float(grid[ei]), float(grid[ei + 1]),
                                     xtol=1e-9, rtol=1e-10))
        out.append((a, b))
    return out
