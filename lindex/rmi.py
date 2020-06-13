"""Two-stage recursive model index (Kraska et al., SIGMOD 2018)."""

import math

import numpy as np

from .base import Index


def fit_linear(x, y):
    """Least-squares fit ``y ~ slope * x + intercept`` (float64 arrays)."""
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return 0.0, float(y[0])
    xm = x.mean()
    ym = y.mean()
    dx = x - xm
    var = float(np.dot(dx, dx))
    if var == 0.0:
        return 0.0, float(ym)
    slope = float(np.dot(dx, y - ym)) / var
    return slope, float(ym - slope * xm)


class RMI(Index):
    """Two-stage RMI with linear models in both stages.

    The root model maps a key to one of ``num_leaves`` second-stage models.
    Each leaf is a linear model that predicts the key's position and stores
    the smallest and largest prediction error seen on its training keys. A
    lookup evaluates both models and finishes with a binary search inside
    the error window (the "last-mile" search).

    Because the root model is monotone, keys are routed to leaves in sorted
    order and every leaf owns a contiguous slice of the key array.
    """

    name = "RMI"

    def __init__(self, num_leaves=1024):
        super().__init__()
        if num_leaves < 1:
            raise ValueError("num_leaves must be positive")
        self.num_leaves = num_leaves
        self._ra = self._rb = 0.0
        self._slopes = []
        self._icpts = []
        self._err_lo = []
        self._err_hi = []
        self._bounds = [0, 0]

    def build(self, keys):
        self._set_keys(keys)
        n = len(self._keys)
        m = self.num_leaves
        self._slopes = [0.0] * m
        self._icpts = [0.0] * m
        self._err_lo = [0] * m
        self._err_hi = [0] * m
        self._ra = self._rb = 0.0
        self._bounds = [0] * (m + 1)
        if n == 0:
            return self

        x = np.asarray(self._keys, dtype=np.float64)
        y = np.arange(n, dtype=np.float64)

        # Root: predict the position, then scale it to a leaf id.
        root_slope, root_icpt = fit_linear(x, y)
        self._ra = root_slope * m / n
        self._rb = root_icpt * m / n
        routes = np.clip(np.floor(self._ra * x + self._rb), 0, m - 1).astype(np.int64)
        bounds = np.searchsorted(routes, np.arange(m + 1), side="left")

        for j in range(m):
            s, e = int(bounds[j]), int(bounds[j + 1])
            if s == e:
                # empty leaf: every key routed here has lower bound s
                self._icpts[j] = float(s)
                continue
            slope, icpt = fit_linear(x[s:e], y[s:e])
            err = y[s:e] - (slope * x[s:e] + icpt)
            self._slopes[j] = slope
            self._icpts[j] = icpt
            self._err_lo[j] = int(math.floor(err.min()))
            self._err_hi[j] = int(math.ceil(err.max()))
        self._bounds = [int(b) for b in bounds]
        return self

    def search_window(self, key):
        m = self.num_leaves
        j = int(self._ra * key + self._rb)
        if j < 0:
            j = 0
        elif j >= m:
            j = m - 1
        pred = self._slopes[j] * key + self._icpts[j]
        # int() rounds towards zero, which is a valid bound in both directions
        lo = int(pred + self._err_lo[j])
        hi = int(pred + self._err_hi[j]) + 1
        # The answer always lies inside the leaf's slice (routing is
        # monotone), which also bounds the search for keys that were not in
        # the training set and for leaves with poorly fitting models.
        start = self._bounds[j]
        end = self._bounds[j + 1]
        if lo < start:
            lo = start
        if hi > end:
            hi = end
        if lo >= hi:
            lo, hi = start, end
        return lo, hi

    def max_error(self):
        """Largest last-mile search window over all leaves."""
        if not self._err_hi:
            return 0
        windows = []
        for j, (l, h) in enumerate(zip(self._err_lo, self._err_hi)):
            windows.append(min(h - l, self._bounds[j + 1] - self._bounds[j]))
        return max(windows)

    def size_bytes(self):
        # root model + (slope, intercept, err_lo, err_hi, start) per leaf
        return 16 + 40 * self.num_leaves

    def params(self):
        return {"num_leaves": self.num_leaves}
