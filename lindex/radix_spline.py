"""RadixSpline: a single-pass learned index (Kipf et al., aiDM 2020).

The index has two parts:

* an error-bounded linear spline through a subset of the (key, position)
  points, computed in one pass by GreedySplineCorridor (Neumann and Michel,
  "Smooth interpolating histograms with error guarantees", BNCOD 2008);
* a radix table over the ``radix_bits`` most significant bits of
  ``key - min_key`` that maps a key prefix to the range of spline points
  sharing that prefix, which narrows the search for the spline segment.

A lookup reads two radix table entries, binary searches a (usually tiny)
range of spline points, interpolates, and finishes with a bounded search
of at most ``2 * eps + 3`` keys.
"""

from bisect import bisect_left

from .base import Index


def greedy_spline_corridor(keys, eps):
    """Spline points through ``(keys[i], i)`` with interpolation error <= eps.

    Every spline point is a data point. The corridor is the cone of lines
    from the last spline point that stay within ``eps`` of all points seen
    since; when a point leaves the corridor, its predecessor becomes the
    next spline point. All arithmetic is on integers, so the test against
    the corridor is exact.
    """
    n = len(keys)
    if n == 0:
        return [], []
    xs, ys = [keys[0]], [0]
    if n == 1:
        return xs, ys
    bx, by = keys[0], 0  # base: the last spline point
    ux, uy = keys[1], 1 + eps  # upper limit of the corridor
    lx, ly = keys[1], 1 - eps  # lower limit of the corridor
    px, py = keys[1], 1  # previous point
    for y in range(2, n):
        x = keys[y]
        dx = x - bx
        dy = y - by
        # slope(base -> p) > slope(base -> upper), or < slope(base -> lower)
        if dy * (ux - bx) > (uy - by) * dx or dy * (lx - bx) < (ly - by) * dx:
            xs.append(px)
            ys.append(py)
            bx, by = px, py
            ux, uy = x, y + eps
            lx, ly = x, y - eps
        else:
            if (y + eps - by) * (ux - bx) < (uy - by) * dx:
                ux, uy = x, y + eps
            if (y - eps - by) * (lx - bx) > (ly - by) * dx:
                lx, ly = x, y - eps
        px, py = x, y
    xs.append(px)
    ys.append(py)
    return xs, ys


class RadixSpline(Index):
    name = "RadixSpline"

    def __init__(self, eps=32, radix_bits=18):
        super().__init__()
        if eps < 1:
            raise ValueError("eps must be at least 1")
        if not 0 <= radix_bits <= 30:
            raise ValueError("radix_bits must be in [0, 30]")
        self.eps = eps
        self.radix_bits = radix_bits
        self._xs = []
        self._ys = []
        self._table = [0, 0]
        self._shift = 0
        self._min = 0

    def build(self, keys):
        self._set_keys(keys)
        keys = self._keys
        self._xs, self._ys = greedy_spline_corridor(keys, self.eps)
        if not keys:
            self._table = [0, 0]
            return self
        self._min = keys[0]
        key_range = keys[-1] - keys[0]
        self._shift = max(key_range.bit_length() - self.radix_bits, 0)
        num_prefixes = (key_range >> self._shift) + 1

        # table[p] = index of the first spline point whose prefix is >= p
        table = [0] * (num_prefixes + 1)
        xs, mn, shift = self._xs, self._min, self._shift
        m = len(xs)
        j = 0
        for p in range(num_prefixes + 1):
            while j < m and (xs[j] - mn) >> shift < p:
                j += 1
            table[p] = j
        self._table = table
        return self

    def search_window(self, key):
        keys = self._keys
        n = len(keys)
        if n == 0 or key <= keys[0]:
            return 0, 0
        if key > keys[-1]:
            return n, n
        xs, ys = self._xs, self._ys
        p = (key - self._min) >> self._shift
        # first spline point >= key; it has prefix p, or is the first one
        # with a larger prefix (table[p + 1])
        i = bisect_left(xs, key, self._table[p], self._table[p + 1])
        if xs[i] == key:
            return ys[i], ys[i]
        x0, y0, x1, y1 = xs[i - 1], ys[i - 1], xs[i], ys[i]
        pred = y0 + (key - x0) * (y1 - y0) / (x1 - x0)
        p = int(pred)
        eps = self.eps
        lo = p - eps - 1
        hi = p + eps + 2
        # key lies strictly between two spline points
        if lo <= y0:
            lo = y0 + 1
        if hi > y1:
            hi = y1
        return lo, hi

    def num_spline_points(self):
        return len(self._xs)

    def size_bytes(self):
        # (key, position) per spline point + 4-byte radix table entries
        return 16 * len(self._xs) + 4 * len(self._table)

    def params(self):
        return {"eps": self.eps, "radix_bits": self.radix_bits}
