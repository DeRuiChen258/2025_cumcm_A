import numpy as np

from src.kinematics import burst_point, shell_position, uav_position


def test_shell_vertical_drop():
    drop = np.array([0.0, 0.0, 1800.0])
    p = shell_position(drop, 0.0, 0.0, 3.6)
    assert np.isclose(p[2], 1800.0 - 0.5 * 9.8 * 3.6 ** 2)


def test_uav_straight_line():
    p = uav_position(np.array([17800.0, 0.0, 1800.0]), np.pi, 120.0, 1.5)
    assert np.allclose(p, [17620.0, 0.0, 1800.0])


def test_burst_point_matches_shell_position():
    drop = np.array([1.0, 2.0, 100.0])
    b = burst_point(drop, 0.3, 90.0, 4.0)
    assert np.allclose(b, shell_position(drop, 0.3, 90.0, 4.0))
