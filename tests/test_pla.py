from bisect import bisect_left

import pytest

from lindex import PLAIndex
from lindex.pla import shrinking_cone_segments


@pytest.mark.parametrize("eps,eps_internal", [(1, 1), (4, 2), (64, 4), (1024, 16)])
def test_lower_bound_matches_bisect(eps, eps_internal, dataset, check_lower_bound):
    _, keys = dataset
    check_lower_bound(PLAIndex(eps, eps_internal).build(keys), keys)


@pytest.mark.parametrize("keys", [[], [42], [1, 2], [3, 7, 9], list(range(0, 500, 7))])
def test_small_inputs(keys):
    index = PLAIndex(eps=1, eps_internal=1).build(keys)
    for q in range(-3, 600):
        assert index.lower_bound(q) == bisect_left(keys, q)


@pytest.mark.parametrize("eps", [1, 8, 64])
def test_segments_respect_error_bound(eps, dataset):
    _, keys = dataset
    first_x, slopes, first_y = shrinking_cone_segments(keys, eps)
    assert first_x[0] == keys[0] and first_y[0] == 0
    assert first_y == sorted(first_y)
    bounds = first_y[1:] + [len(keys)]
    for x0, slope, start, end in zip(first_x, slopes, first_y, bounds):
        assert end - start >= 1
        for pos in range(start, end):
            pred = start + slope * (keys[pos] - x0)
            assert abs(pred - pos) <= eps + 1e-6


def test_linear_data_needs_one_segment():
    keys = list(range(0, 10 ** 6, 10))
    index = PLAIndex(eps=1).build(keys)
    assert index.num_segments() == 1
    assert index.height() == 1


def test_larger_eps_means_fewer_segments(dataset):
    _, keys = dataset
    counts = [PLAIndex(eps).build(keys).num_segments() for eps in (2, 16, 128)]
    assert counts[0] > counts[1] > counts[2]


def test_levels_shrink_geometrically(dataset):
    _, keys = dataset
    index = PLAIndex(eps=2, eps_internal=1).build(keys)
    sizes = [len(level[0]) for level in index._levels]
    assert sizes[-1] == 1
    assert all(upper <= (lower + 1) // 2 for lower, upper in zip(sizes, sizes[1:]))


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        PLAIndex(eps=0)
    with pytest.raises(ValueError):
        PLAIndex(eps=4, eps_internal=0)
