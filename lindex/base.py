"""Interface shared by all index implementations."""

from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from typing import List, Sequence


class Index(ABC):
    """A read-only index over a sorted array of unique integer keys.

    Every index answers ``lower_bound(key)``: the position of the first key
    in the indexed array that is not less than ``key``. This mirrors
    ``std::lower_bound`` and the lookup semantics used by SOSD, so an index
    can be checked against plain binary search for arbitrary probe keys,
    not only for keys that are present.
    """

    #: Short name used in benchmark output.
    name = "Index"

    def __init__(self):
        self._keys = []  # type: List[int]

    @abstractmethod
    def build(self, keys: Sequence[int]) -> "Index":
        """Build the index over ``keys`` (sorted, unique) and return ``self``."""

    @abstractmethod
    def lower_bound(self, key: int) -> int:
        """Return the position of the first key that is >= ``key``."""

    @abstractmethod
    def size_bytes(self) -> int:
        """Approximate size of the index structure, excluding the keys."""

    def params(self) -> dict:
        """Tuning parameters, reported next to the benchmark numbers."""
        return {}

    def contains(self, key: int) -> bool:
        pos = self.lower_bound(key)
        return pos < len(self._keys) and self._keys[pos] == key

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        args = ", ".join("{}={}".format(k, v) for k, v in self.params().items())
        return "{}({})".format(type(self).__name__, args)

    def _set_keys(self, keys: Sequence[int]) -> None:
        # bisect is fastest on plain lists, so avoid keeping numpy arrays.
        self._keys = keys if isinstance(keys, list) else [int(k) for k in keys]


def bounded_lower_bound(keys: List[int], key: int, lo: int, hi: int) -> int:
    """Binary search for ``key`` in ``keys[lo:hi]`` and validate the result.

    Learned indexes only guarantee their error bound for keys that were
    present at build time. For other keys the predicted window may miss the
    true position, so the result is checked against the neighbouring
    elements and we fall back to a full binary search when needed.
    """
    n = len(keys)
    if lo < 0:
        lo = 0
    elif lo > n:
        lo = n
    if hi < lo:
        hi = lo
    elif hi > n:
        hi = n
    pos = bisect_left(keys, key, lo, hi)
    if (lo > 0 and keys[lo - 1] >= key) or (pos == hi and hi < n and keys[hi] < key):
        return bisect_left(keys, key)
    return pos


def bounded_upper_bound(keys: List[int], key: int, lo: int, hi: int) -> int:
    """Like :func:`bounded_lower_bound`, but for ``bisect_right``."""
    n = len(keys)
    if lo < 0:
        lo = 0
    elif lo > n:
        lo = n
    if hi < lo:
        hi = lo
    elif hi > n:
        hi = n
    pos = bisect_right(keys, key, lo, hi)
    if (lo > 0 and keys[lo - 1] > key) or (pos == hi and hi < n and keys[hi] <= key):
        return bisect_right(keys, key)
    return pos
