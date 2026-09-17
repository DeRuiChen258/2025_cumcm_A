"""GPU batched differential evolution for Q5 using the Triton objective."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .config import SEED
from .triton_objective import q5_batch_durations_triton


def run_gpu_de_q5(bounds: Sequence[Tuple[float, float]], popsize: int,
                  maxiter: int, seed: int = SEED,
                  init: Optional[np.ndarray] = None,
                  f_mutation: float = 0.75, cr: float = 0.9) -> Tuple[np.ndarray, float, np.ndarray]:
    """Batched rand/1/bin DE; each generation evaluates the whole population in one GPU launch."""
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    dim = len(bounds)
    p_count = popsize * max(1, dim - 1)
    rng = np.random.default_rng(seed)

    if init is None or len(init) < p_count:
        rand_pop = rng.uniform(lo, hi, size=(p_count, dim))
        if init is None:
            pop = rand_pop
        else:
            pop = np.vstack([np.asarray(init, dtype=np.float64), rand_pop])[:p_count]
    else:
        pop = np.asarray(init[:p_count], dtype=np.float64).copy()

    dur = q5_batch_durations_triton(pop)
    energies = -dur.sum(axis=1)
    best_i = int(np.argmin(energies))
    best_x = pop[best_i].copy()
    best_e = float(energies[best_i])
    history = [best_e]
    idx_all = np.arange(p_count)

    for it in range(maxiter):
        # vectorized distinct random indices (resample collisions)
        a = rng.integers(0, p_count, size=p_count)
        b = rng.integers(0, p_count, size=p_count)
        c = rng.integers(0, p_count, size=p_count)
        bad = (a == idx_all) | (b == idx_all) | (c == idx_all) | (a == b) | (a == c) | (b == c)
        guard = 0
        while bad.any() and guard < 20:
            n_bad = int(bad.sum())
            a[bad] = rng.integers(0, p_count, size=n_bad)
            b[bad] = rng.integers(0, p_count, size=n_bad)
            c[bad] = rng.integers(0, p_count, size=n_bad)
            bad = (a == idx_all) | (b == idx_all) | (c == idx_all) | (a == b) | (a == c) | (b == c)
            guard += 1
        idxs = np.vstack([a, b, c])
        a, b, c = idxs
        mutant = pop[a] + f_mutation * (pop[b] - pop[c])
        mutant = np.clip(mutant, lo, hi)
        cross = rng.random((p_count, dim)) < cr
        jrand = rng.integers(0, dim, size=p_count)
        mask = cross | (np.arange(dim)[None, :] == jrand[:, None])
        trial = np.where(mask, mutant, pop)

        t_dur = q5_batch_durations_triton(trial)
        t_energy = -t_dur.sum(axis=1)
        better = t_energy < energies
        pop[better] = trial[better]
        energies[better] = t_energy[better]

        cand_i = int(np.argmin(t_energy))
        if t_energy[cand_i] < best_e:
            best_x = trial[cand_i].copy()
            best_e = float(t_energy[cand_i])
        # elite preservation
        pop[0] = best_x
        energies[0] = best_e
        history.append(best_e)
    return best_x, best_e, np.asarray(history)
