import pytest

from lindex.datasets import DISTRIBUTIONS, MAX_KEY, generate, load_sosd, write_sosd


@pytest.mark.parametrize("name", DISTRIBUTIONS)
def test_generate_sorted_unique(name):
    keys = generate(name, 5000, seed=1)
    assert len(keys) == 5000
    assert all(a < b for a, b in zip(keys, keys[1:]))
    assert keys[0] >= 0 and keys[-1] < MAX_KEY
    assert all(isinstance(k, int) for k in keys[:10])


@pytest.mark.parametrize("name", DISTRIBUTIONS)
def test_generate_is_deterministic(name):
    assert generate(name, 1000, seed=3) == generate(name, 1000, seed=3)
    assert generate(name, 1000, seed=3) != generate(name, 1000, seed=4)


def test_generate_empty():
    assert generate("uniform", 0) == []


def test_unknown_distribution():
    with pytest.raises(ValueError):
        generate("zipf", 10)


def test_sosd_roundtrip(tmp_path):
    keys = generate("lognormal", 2000, seed=5)
    path = str(tmp_path / "keys_uint64")
    write_sosd(path, keys)
    assert load_sosd(path) == keys
    assert load_sosd(path, limit=100) == keys[:100]


def test_sosd_loader_sorts_and_dedups(tmp_path):
    path = str(tmp_path / "dups_uint64")
    write_sosd(path, [5, 3, 3, 9, 1])
    assert load_sosd(path) == [1, 3, 5, 9]
