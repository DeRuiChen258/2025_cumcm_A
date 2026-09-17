import numpy as np

from src.objective import decode_q3, decode_q5


def test_q3_spacing():
    x = [0.0, 100.0, 1.0, 5.0, 2.0, 3.0, 4.0, 5.0]
    s = decode_q3(x)
    tds = [sh["t_d"] for sh in s["shells"]]
    tds.sort()
    assert tds[1] - tds[0] >= 1.0 - 1e-9
    assert tds[2] - tds[1] >= 1.0 - 1e-9


def test_q5_spacing_per_uav():
    x = np.array([0.0, 100.0, 1.0, 5.0, 2.0, 3.0, 4.0, 5.0] * 5)
    s = decode_q5(x)
    for k in range(5):
        tds = sorted(sh["t_d"] for sh in s["shells"] if sh["uav"] == k)
        assert tds[1] - tds[0] >= 1.0 - 1e-9
        assert tds[2] - tds[1] >= 1.0 - 1e-9
