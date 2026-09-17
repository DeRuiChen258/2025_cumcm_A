"""Differential evolution + local refinement helpers."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .config import SEED


def run_de(objective: Callable[[np.ndarray], float], bounds: Sequence[Tuple[float, float]],
           popsize: int, maxiter: int, seed: int = SEED, workers: int = -1,
           polish: bool = True,
           init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, List[float]]:
    """Run differential evolution, returning (x, fun, best-history per iteration)."""
    history: List[float] = []
    n_iter = {"n": 0}

    def callback(xk: np.ndarray, convergence: float) -> None:
        n_iter["n"] += 1
        history.append(float(objective(np.asarray(xk, dtype=float))))

    res = differential_evolution(
        lambda x: float(objective(np.asarray(x, dtype=float))),
        bounds,
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-9,
        polish=polish,
        workers=1,
        updating="immediate",
        callback=callback,
        init=init,
    )
    return np.asarray(res.x, dtype=float), float(res.fun), history


def local_refine(objective: Callable[[np.ndarray], float], x0: np.ndarray,
                 bounds: Sequence[Tuple[float, float]],
                 maxiter: int = 3000) -> Tuple[np.ndarray, float]:
    """L-BFGS-B then bounded Nelder-Mead refinement."""
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)

    def obj(x: np.ndarray) -> float:
        return float(objective(np.clip(np.asarray(x, dtype=float), lo, hi)))

    best_x = x0.copy()
    best_f = obj(best_x)

    res1 = minimize(obj, x0, method="L-BFGS-B", bounds=list(bounds),
                    options={"ftol": 1e-12, "maxiter": maxiter})
    if res1.fun < best_f:
        best_x, best_f = np.asarray(res1.x, dtype=float), float(res1.fun)

    res2 = minimize(obj, best_x, method="Nelder-Mead",
                    options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-11})
    if res2.fun < best_f:
        best_x, best_f = np.asarray(res2.x, dtype=float), float(res2.fun)
    return np.clip(best_x, lo, hi), best_f


def multistart(objective: Callable[[np.ndarray], float],
               bounds: Sequence[Tuple[float, float]], n_starts: int = 32,
               seed: int = SEED) -> Tuple[np.ndarray, float]:
    """Random multi-start local search (used for Q2 global check)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    starts = rng.uniform(lo, hi, size=(n_starts, len(bounds)))
    starts[0] = (lo + hi) / 2.0
    best_x, best_f = None, np.inf
    for x0 in starts:
        x, f = local_refine(objective, x0, bounds, maxiter=1200)
        if f < best_f:
            best_x, best_f = x.copy(), f
    return best_x, best_f
