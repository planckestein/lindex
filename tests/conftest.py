import random
from bisect import bisect_left

import pytest

from lindex.datasets import DISTRIBUTIONS, generate

NUM_TEST_KEYS = 20000


@pytest.fixture(scope="session", params=DISTRIBUTIONS)
def dataset(request):
    return request.param, generate(request.param, NUM_TEST_KEYS, seed=7)


def _probes(keys, num, seed):
    """Present keys, their neighbours, random gaps and out-of-range keys."""
    rng = random.Random(seed)
    probes = [keys[0] - 1, keys[0], keys[-1], keys[-1] + 1, -(2 ** 40), 2 ** 62]
    for _ in range(num):
        k = keys[rng.randrange(len(keys))]
        probes.extend((k, k - 1, k + 1))
    probes.extend(rng.randint(keys[0], keys[-1]) for _ in range(num))
    return probes


@pytest.fixture
def check_lower_bound():
    """Compare ``index.lower_bound`` with ``bisect_left`` on many probe keys."""

    def check(index, keys, num_probes=3000, seed=0):
        for q in _probes(keys, num_probes, seed):
            expected = bisect_left(keys, q)
            got = index.lower_bound(q)
            assert got == expected, "{!r}: lower_bound({}) = {}, expected {}".format(
                index, q, got, expected
            )

    return check
