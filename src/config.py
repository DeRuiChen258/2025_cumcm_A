"""Global constants and input data for the smoke decoy problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# --- physics ---
G = 9.8
MISSILE_SPEED = 300.0
CLOUD_RADIUS = 10.0
SINK_SPEED = 3.0
CLOUD_LIFETIME = 20.0
MIN_BURST_HEIGHT = 60.0

# --- target ---
TARGET_CENTER = np.array([0.0, 200.0, 0.0])
CYLINDER_RADIUS = 7.0
CYLINDER_HEIGHT = 10.0

# --- missiles / UAVs ---
MISSILES: Dict[str, np.ndarray] = {
    "M1": np.array([20000.0, 0.0, 2000.0]),
    "M2": np.array([19000.0, 600.0, 2100.0]),
    "M3": np.array([18000.0, -600.0, 1900.0]),
}

UAVS: Dict[str, np.ndarray] = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
    "FY4": np.array([11000.0, 2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0]),
}

UAV_NAMES = list(UAVS.keys())
MISSILE_NAMES = list(MISSILES.keys())

# --- optimization ---
SEED = 2025
SAMPLE_DT = 0.1
UAV_SPEED_RANGE = (70.0, 140.0)
DROP_TIME_RANGE = (0.0, 55.0)
TAU_MIN = 0.5


@dataclass(frozen=True)
class MissileSpec:
    name: str
    p0: np.ndarray
    direction: np.ndarray
    impact_time: float


@dataclass(frozen=True)
class UavSpec:
    name: str
    p0: np.ndarray
    height: float
    tau_max: float


def missile_spec(name: str) -> MissileSpec:
    p0 = MISSILES[name].astype(float)
    dist = float(np.linalg.norm(p0))
    direction = -p0 / dist
    return MissileSpec(name=name, p0=p0, direction=direction,
                       impact_time=dist / MISSILE_SPEED)


def uav_spec(name: str) -> UavSpec:
    p0 = UAVS[name].astype(float)
    h = float(p0[2])
    tau_max = float(np.sqrt(max(2.0 * (h - MIN_BURST_HEIGHT) / G, 0.0)))
    return UavSpec(name=name, p0=p0, height=h, tau_max=tau_max)


def tau_bounds(uav_name: str) -> Tuple[float, float]:
    return TAU_MIN, uav_spec(uav_name).tau_max
