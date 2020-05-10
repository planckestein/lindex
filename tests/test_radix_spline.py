from bisect import bisect_left

import pytest

from lindex import RadixSpline
from lindex.radix_spline import greedy_spline_corridor


@pytest.mark.parametrize("eps,radix_bits", [(1, 0), (4, 8), (32, 18), (256, 20)])
def test_lower_bound_matches_bisect(eps, radix_bits, dataset, check_lower_bound):
    _, keys = dataset
    check_lower_bound(RadixSpline(eps, radix_bits).build(keys), keys)


@pytest.mark.parametrize("keys", [[], [42], [1, 2], [3, 7, 9], list(range(0, 500, 7))])
@pytest.mark.parametrize("radix_bits", [0, 2, 16])
def test_small_inputs(keys, radix_bits):
    index = RadixSpline(eps=1, radix_bits=radix_bits).build(keys)
    for q in range(-3, 600):
        assert index.lower_bound(q) == bisect_left(keys, q)


@pytest.mark.parametrize("eps", [1, 8, 64])
def test_spline_respects_error_bound(eps, dataset):
    _, keys = dataset
    xs, ys = greedy_spline_corridor(keys, eps)
    assert xs[0] == keys[0] and xs[-1] == keys[-1]
    assert all(keys[y] == x for x, y in zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        for pos in range(y0, y1 + 1):
            pred = y0 + (keys[pos] - x0) * (y1 - y0) / (x1 - x0)
            assert abs(pred - pos) <= eps + 1e-6


def test_linear_data_needs_two_spline_points():
    keys = list(range(0, 10 ** 6, 10))
    assert greedy_spline_corridor(keys, 1) == ([0, keys[-1]], [0, len(keys) - 1])


def test_radix_table_is_monotone(dataset):
    _, keys = dataset
    index = RadixSpline(eps=16, radix_bits=12).build(keys)
    table = index._table
    assert table[0] == 0
    assert table[-1] == index.num_spline_points()
    assert all(a <= b for a, b in zip(table, table[1:]))
    assert len(table) <= 2 ** 12 + 1


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        RadixSpline(eps=0)
    with pytest.raises(ValueError):
        RadixSpline(radix_bits=31)
