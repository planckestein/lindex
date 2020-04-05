from bisect import bisect_left

import numpy as np
import pytest

from lindex import RMI
from lindex.rmi import fit_linear


@pytest.mark.parametrize("num_leaves", [1, 16, 1024, 2 ** 14])
def test_lower_bound_matches_bisect(num_leaves, dataset, check_lower_bound):
    _, keys = dataset
    check_lower_bound(RMI(num_leaves).build(keys), keys)


@pytest.mark.parametrize("keys", [[], [42], [1, 2], [0, 1000, 10 ** 12]])
def test_small_inputs(keys):
    index = RMI(num_leaves=8).build(keys)
    for q in [-1, 0, 1, 2, 41, 42, 43, 999, 1000, 1001, 10 ** 12, 10 ** 13]:
        assert index.lower_bound(q) == bisect_left(keys, q)


def test_error_bounds_hold_for_training_keys():
    keys = list(range(0, 300000, 3))
    index = RMI(num_leaves=64).build(keys)
    # perfectly linear data: every leaf model should be (almost) exact
    assert index.max_error() <= 2
    for pos in range(0, len(keys), 997):
        assert index.lower_bound(keys[pos]) == pos


def test_more_leaves_shrink_error(dataset):
    _, keys = dataset
    small = RMI(num_leaves=16).build(keys).max_error()
    large = RMI(num_leaves=4096).build(keys).max_error()
    assert large <= small


def test_fit_linear():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    slope, icpt = fit_linear(x, 2 * x + 1)
    assert slope == pytest.approx(2.0)
    assert icpt == pytest.approx(1.0)
    assert fit_linear(np.array([]), np.array([])) == (0.0, 0.0)
    assert fit_linear(np.array([5.0]), np.array([3.0])) == (0.0, 3.0)
    assert fit_linear(np.array([5.0, 5.0]), np.array([1.0, 3.0])) == (0.0, 2.0)


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        RMI(num_leaves=0)
