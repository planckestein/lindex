"""The predicted window must contain the answer, without relying on the
fallback in ``bounded_lower_bound``."""

import random
from bisect import bisect_left

import pytest

from lindex import BinarySearch, BTree, PLAIndex, RadixSpline, RMI

FACTORIES = [
    BinarySearch,
    lambda: BTree(leaf_size=32, fanout=8),
    lambda: RMI(num_leaves=1024),
    lambda: PLAIndex(eps=16, eps_internal=2),
    lambda: RadixSpline(eps=16, radix_bits=12),
]


@pytest.mark.parametrize("factory", FACTORIES)
def test_window_contains_answer(factory, dataset):
    _, keys = dataset
    index = factory().build(keys)
    rng = random.Random(1)
    probes = []
    for _ in range(3000):
        k = keys[rng.randrange(len(keys))]
        probes.extend((k, k + 1))
    for q in probes:
        lo, hi = index.search_window(q)
        assert 0 <= lo <= hi <= len(keys)
        assert lo <= bisect_left(keys, q) <= hi, (index, q, lo, hi)


@pytest.mark.parametrize("eps", [4, 64])
@pytest.mark.parametrize("cls", [PLAIndex, RadixSpline])
def test_window_size_is_bounded_by_eps(cls, eps, dataset):
    _, keys = dataset
    index = cls(eps=eps).build(keys)
    for q in keys[::37]:
        lo, hi = index.search_window(q)
        assert hi - lo <= 2 * eps + 3
