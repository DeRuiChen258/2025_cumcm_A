"""Kinematic models for missiles, UAVs, shells and smoke clouds."""

from __future__ import annotations

import numpy as np

from .config import G, MISSILE_SPEED, SINK_SPEED, TARGET_CENTER


def missile_position(p0: np.ndarray, direction: np.ndarray, t: float) -> np.ndarray:
    return p0 + MISSILE_SPEED * t * direction


def uav_position(p0: np.ndarray, theta: float, v: float, t: float) -> np.ndarray:
    heading = np.array([np.cos(theta), np.sin(theta), 0.0])
    return p0 + v * t * heading


def shell_position(drop_point: np.ndarray, theta: float, v: float, t_rel: float,
                   g_override: float = G) -> np.ndarray:
    """Shell position t_rel seconds after release (no air drag, zero vertical v0)."""
    heading = np.array([np.cos(theta), np.sin(theta), 0.0])
    return drop_point + v * t_rel * heading + np.array([0.0, 0.0, -0.5 * g_override * t_rel ** 2])


def burst_point(drop_point: np.ndarray, theta: float, v: float, tau: float,
                g_override: float = G) -> np.ndarray:
    return shell_position(drop_point, theta, v, tau, g_override=g_override)


def cloud_center(burst: np.ndarray, t_since_burst: float) -> np.ndarray:
    return burst + np.array([0.0, 0.0, -SINK_SPEED * t_since_burst])


def target_point() -> np.ndarray:
    return TARGET_CENTER
