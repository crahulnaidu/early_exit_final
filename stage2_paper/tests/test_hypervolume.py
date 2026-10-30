import numpy as np
import pytest

from utils.hypervolume import hv2d, hv3d, pareto_mask


def test_known_values():
    assert hv2d([[0, 0]], [1, 1]) == 1.0
    assert hv2d([[0, 0.5], [0.5, 0]], [1, 1]) == pytest.approx(0.75)
    assert hv3d([[0, 0, 0]], [1, 2, 3]) == 6.0
    assert hv3d([[2, 2, 2]], [1, 1, 1]) == 0.0


def test_against_optuna_and_monte_carlo():
    from optuna._hypervolume import compute_hypervolume
    rng = np.random.default_rng(0)
    for _ in range(20):
        pts = rng.random((30, 3))
        ref = np.array([1.1, 1.2, 1.3])
        ours = hv3d(pts, ref)
        assert ours == pytest.approx(compute_hypervolume(pts, ref), rel=1e-9)
    pts = rng.random((8, 3))
    ref = np.ones(3)
    samples = rng.random((200_000, 3))
    dominated = np.zeros(len(samples), dtype=bool)
    for p in pts:
        dominated |= np.all(samples >= p, axis=1)
    assert hv3d(pts, ref) == pytest.approx(dominated.mean(), abs=0.01)


def test_pareto_mask():
    pts = [[1, 1, 1], [0, 2, 2], [2, 2, 2], [1, 1, 1]]
    assert pareto_mask(pts).tolist() == [True, True, False, True]
