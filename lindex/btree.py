"""Static, bulk-loaded B+-tree; the classic comparison-based baseline."""

from bisect import bisect_left, bisect_right

from .base import Index


class BTree(Index):
    """A read-only B+-tree built bottom-up over the sorted key array.

    The key array itself acts as the leaf level, split into pages of
    ``leaf_size`` keys. Every inner level stores the first key of each node
    one level below, grouped into nodes of ``fanout`` separators. Levels are
    kept as flat lists, so node ``i`` of a level covers the separators
    ``level[i * fanout:(i + 1) * fanout]`` and no child pointers are needed.
    ``size_bytes`` still charges 8 bytes per pointer to stay comparable with
    a pointer-based implementation.
    """

    name = "BTree"

    def __init__(self, leaf_size=64, fanout=16):
        super().__init__()
        if leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        if fanout < 2:
            raise ValueError("fanout must be at least 2")
        self.leaf_size = leaf_size
        self.fanout = fanout
        self._levels = []

    def build(self, keys):
        self._set_keys(keys)
        seps = self._keys[:: self.leaf_size]
        levels = [seps]
        while len(seps) > self.fanout:
            seps = seps[:: self.fanout]
            levels.append(seps)
        self._levels = levels[::-1]  # root first
        return self

    def lower_bound(self, key):
        keys = self._keys
        if not keys:
            return 0
        fanout = self.fanout
        node = 0
        for level in self._levels:
            lo = node * fanout
            hi = min(lo + fanout, len(level))
            # last separator <= key; clamp to the first child for small keys
            pos = bisect_right(level, key, lo, hi) - 1
            node = pos if pos >= lo else lo
        lo = node * self.leaf_size
        return bisect_left(keys, key, lo, min(lo + self.leaf_size, len(keys)))

    def height(self):
        return len(self._levels)

    def size_bytes(self):
        # one key and one child pointer per separator
        return 16 * sum(len(level) for level in self._levels)

    def params(self):
        return {"leaf_size": self.leaf_size, "fanout": self.fanout}
