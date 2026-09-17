"""Geometric primitives: point-segment distance and interval algebra."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def dist_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to segment [a, b].

    points: (..., 3), a/b: (3,). Returns (...) array.
    """
    points = np.asarray(points, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ba = b - a
    denom = np.einsum("...i,...i->...", ba, ba)
    t = np.einsum("...i,...i->...", points - a, ba) / np.where(denom == 0.0, 1.0, denom)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ba
    return np.linalg.norm(points - proj, axis=-1)


def union_intervals(intervals: Iterable[Tuple[float, float]],
                    eps: float = 1e-9) -> List[Tuple[float, float]]:
    """Merge intervals [a,b] (a <= b), keeping disjoint union length intact."""
    items = sorted((float(a), float(b)) for a, b in intervals if b - a > eps)
    if not items:
        return []
    merged: List[Tuple[float, float]] = [items[0]]
    for a, b in items[1:]:
        ca, cb = merged[-1]
        if a <= cb + 1e-9:
            merged[-1] = (ca, max(cb, b))
        else:
            merged.append((a, b))
    return merged


def total_length(intervals: Sequence[Tuple[float, float]]) -> float:
    return float(sum(b - a for a, b in intervals))


def intersect_intervals(a: Sequence[Tuple[float, float]],
                        b: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Intersection of two interval lists."""
    out: List[Tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo + 1e-9:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def intersect_many(sets: Sequence[Sequence[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    if not sets:
        return []
    cur = list(sets[0])
    for nxt in sets[1:]:
        cur = intersect_intervals(cur, nxt)
        if not cur:
            break
    return cur
