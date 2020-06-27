import csv

import pytest

from lindex import BTree, Index, RMI, bench
from lindex.datasets import generate


def test_run_dataset_small(tmp_path):
    keys = generate("normal", 5000, seed=2)
    factories = [lambda: BTree(leaf_size=16), lambda: RMI(num_leaves=64)]
    results = bench.run_dataset("normal", keys, 500, factories, log=lambda s: None)
    assert [r.index for r in results] == ["BTree", "RMI"]
    for r in results:
        assert r.num_keys == 5000
        assert r.lookup_ns > 0
        assert 0 <= r.log2_window <= 5  # log2(16 + 1) for the B-tree

    path = str(tmp_path / "results.csv")
    bench.write_csv(results, path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[1]["params"] == "num_leaves=64"
    assert "| normal | RMI | num_leaves=64 |" in bench.to_markdown(results)


class Broken(Index):
    name = "Broken"

    def build(self, keys):
        self._set_keys(keys)
        return self

    def search_window(self, key):
        return 0, 0

    def lower_bound(self, key):
        return 0

    def size_bytes(self):
        return 0


def test_verification_catches_wrong_results():
    keys = list(range(100))
    with pytest.raises(AssertionError):
        bench.measure(Broken, "linear", keys, [50])
