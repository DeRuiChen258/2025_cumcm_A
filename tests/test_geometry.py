import numpy as np

from src.geometry import dist_to_segment, intersect_many, union_intervals


def test_dist_on_segment():
    d = dist_to_segment(np.array([[0.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0]),
                        np.array([10.0, 0.0, 0.0]))
    assert d[0] == 0.0


def test_dist_beyond_endpoint():
    d = dist_to_segment(np.array([[12.0, 3.0, 0.0]]), np.array([0.0, 0.0, 0.0]),
                        np.array([10.0, 0.0, 0.0]))
    assert np.isclose(d[0], np.sqrt(13.0))


def test_dist_projection():
    d = dist_to_segment(np.array([[5.0, 4.0, 0.0]]), np.array([0.0, 0.0, 0.0]),
                        np.array([10.0, 0.0, 0.0]))
    assert np.isclose(d[0], 4.0)


def test_union_and_intersect():
    u = union_intervals([(0.0, 2.0), (1.5, 3.0), (5.0, 6.0)])
    assert u == [(0.0, 3.0), (5.0, 6.0)]
    inter = intersect_many([[(0.0, 4.0)], [(1.0, 2.0)], [(0.5, 1.5)]])
    assert inter == [(1.0, 1.5)]
