import pytest

from lindex import BinarySearch, BTree


@pytest.mark.parametrize(
    "factory",
    [
        BinarySearch,
        lambda: BTree(),
        lambda: BTree(leaf_size=1, fanout=2),
        lambda: BTree(leaf_size=256, fanout=64),
    ],
)
def test_lower_bound_matches_bisect(factory, dataset, check_lower_bound):
    _, keys = dataset
    check_lower_bound(factory().build(keys), keys)


@pytest.mark.parametrize("factory", [BinarySearch, BTree])
@pytest.mark.parametrize("keys", [[], [42], [1, 2], list(range(0, 1000, 3))])
def test_small_inputs(factory, keys):
    index = factory().build(keys)
    for q in range(-2, 1005):
        expected = sum(1 for k in keys if k < q)
        assert index.lower_bound(q) == expected


def test_btree_height_and_size():
    keys = list(range(100000))
    tree = BTree(leaf_size=64, fanout=16).build(keys)
    # 1563 leaf pages -> 98 -> 7 separators
    assert tree.height() == 3
    assert tree.size_bytes() == 16 * (1563 + 98 + 7)


def test_btree_contains():
    tree = BTree(leaf_size=4, fanout=4).build([2, 4, 8, 16, 32])
    assert tree.contains(8)
    assert not tree.contains(9)
    assert not tree.contains(64)


@pytest.mark.parametrize("kwargs", [{"leaf_size": 0}, {"fanout": 1}])
def test_btree_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        BTree(**kwargs)
