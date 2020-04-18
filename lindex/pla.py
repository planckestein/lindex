"""Piecewise linear approximation (PLA) index with a guaranteed error bound.

Keys are split into segments by the greedy shrinking-cone algorithm of
FITing-Tree (Galakatos et al., SIGMOD 2019). Every segment is a line that
predicts the position of each of its keys within ``eps`` positions, so the
last-mile search touches at most ``2 * eps + 3`` keys.
"""

import math
from bisect import bisect_right

from .base import Index, bounded_lower_bound


def shrinking_cone_segments(xs, eps):
    """Segment the points ``(xs[i], i)`` so that each line errs by <= eps.

    Returns three parallel lists: the first key of each segment, its slope,
    and the position of its first key. A segment starting at ``(x0, y0)``
    keeps the cone of slopes that are still feasible for every point added
    so far; a new segment starts as soon as the cone becomes empty.
    """
    first_x, slopes, first_y = [], [], []
    n = len(xs)
    i = 0
    while i < n:
        x0 = xs[i]
        lo_s, hi_s = 0.0, math.inf
        j = i + 1
        while j < n:
            dx = xs[j] - x0
            dy = j - i
            cand_lo = (dy - eps) / dx
            cand_hi = (dy + eps) / dx
            if cand_lo < lo_s:
                cand_lo = lo_s
            if cand_hi > hi_s:
                cand_hi = hi_s
            if cand_lo > cand_hi:
                break
            lo_s, hi_s = cand_lo, cand_hi
            j += 1
        first_x.append(x0)
        slopes.append(0.0 if hi_s == math.inf else (lo_s + hi_s) / 2.0)
        first_y.append(i)
        i = j
    return first_x, slopes, first_y


def _window(first_x, slopes, first_y, seg, key, eps, size):
    """Search window ``[lo, hi)`` for ``key`` inside segment ``seg``.

    For keys of the segment the prediction is within ``eps``. A key that
    falls between two keys of the segment (or after its last key) has a
    true position at most one further away, hence the ``+ 1`` slack. The
    prediction is clamped to the segment, which is where the answer lies.
    """
    start = first_y[seg]
    end = first_y[seg + 1] if seg + 1 < len(first_y) else size
    pred = start + slopes[seg] * (key - first_x[seg])
    if pred > end:
        pred = end
    p = int(pred)
    lo = p - eps - 1
    hi = p + eps + 2
    if lo < start:
        lo = start
    if hi > end:
        hi = end
    return lo, hi


class PLAIndex(Index):
    """Error-bounded piecewise linear index (single level).

    Segments are located by binary search over their first keys.
    """

    name = "PLA"

    def __init__(self, eps=64):
        super().__init__()
        if eps < 1:
            raise ValueError("eps must be at least 1")
        self.eps = eps
        self._segments = ([], [], [])

    def build(self, keys):
        self._set_keys(keys)
        self._segments = shrinking_cone_segments(self._keys, self.eps)
        return self

    def lower_bound(self, key):
        keys = self._keys
        n = len(keys)
        if n == 0 or key <= keys[0]:
            return 0
        if key > keys[-1]:
            return n
        first_x, slopes, first_y = self._segments
        seg = bisect_right(first_x, key) - 1
        lo, hi = _window(first_x, slopes, first_y, seg, key, self.eps, n)
        return bounded_lower_bound(keys, key, lo, hi)

    def num_segments(self):
        return len(self._segments[0])

    def size_bytes(self):
        # first key, slope and first position per segment
        return 24 * self.num_segments()

    def params(self):
        return {"eps": self.eps}
