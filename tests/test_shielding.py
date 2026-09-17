import numpy as np

from src.config import TARGET_CENTER
from src.shielding import shielded_intervals


def test_must_shield():
    iv = shielded_intervals(np.array([1000.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
                            100.0, np.array([500.0, 0.0, 0.0]), 0.0,
                            target=TARGET_CENTER * 0 + np.array([0.0, 0.0, 0.0]),
                            refine=True)
    assert len(iv) >= 1


def test_must_not_shield():
    iv = shielded_intervals(np.array([1000.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
                            100.0, np.array([500.0, 1000.0, 0.0]), 0.0,
                            target=TARGET_CENTER * 0 + np.array([0.0, 0.0, 0.0]),
                            refine=True)
    assert iv == []


def test_refined_matches_dense():
    p0 = np.array([20000.0, 0.0, 2000.0])
    dist = np.linalg.norm(p0)
    u = -p0 / dist
    burst = np.array([17188.0, 0.0, 1736.5])
    t_e = 5.1
    refined = shielded_intervals(p0, u, dist / 300.0, burst, t_e,
                                 target=TARGET_CENTER, refine=True)
    dense = shielded_intervals(p0, u, dist / 300.0, burst, t_e,
                               target=TARGET_CENTER, refine=False, dt=0.005)
    assert len(refined) == len(dense)
    for (a1, b1), (a2, b2) in zip(refined, dense):
        assert abs(a1 - a2) < 0.01
        assert abs(b1 - b2) < 0.01
