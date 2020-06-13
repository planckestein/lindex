"""Binary search over the key array; the zero-space baseline."""

from bisect import bisect_left

from .base import Index


class BinarySearch(Index):
    """Plain binary search. Uses no extra space beyond the keys."""

    name = "BinarySearch"

    def build(self, keys):
        self._set_keys(keys)
        return self

    def search_window(self, key):
        return 0, len(self._keys)

    def lower_bound(self, key):
        return bisect_left(self._keys, key)

    def size_bytes(self):
        return 0
