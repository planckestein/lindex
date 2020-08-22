"""Key distributions used in the experiments.

All generators return a sorted Python list of unique, non-negative integers.
Keys are kept below 2**53 so that they convert to float64 without loss of
precision, since the learned models are fit in floating point.

Real-world data can be loaded from the binary format used by SOSD
(https://github.com/learnedsystems/SOSD): a little-endian uint64 holding the
number of keys, followed by the keys themselves as little-endian uint64.
"""

import struct
import warnings
from typing import List, Optional, Sequence

import numpy as np

DISTRIBUTIONS = ("uniform", "normal", "lognormal", "clustered")

#: Exclusive upper bound for generated keys (exactly representable as float64).
MAX_KEY = 2 ** 53


def generate(name: str, n: int, seed: int = 42) -> List[int]:
    """Generate ``n`` sorted, unique keys drawn from distribution ``name``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = np.random.default_rng(seed)
    keys = np.empty(0, dtype=np.int64)
    rounds = 0
    while len(keys) < n:
        rounds += 1
        if rounds > 100:
            raise RuntimeError(
                "could not draw {} unique keys from {!r}".format(n, name)
            )
        need = n - len(keys)
        keys = np.union1d(keys, _sample(name, rng, int(need * 1.1) + 16))
    if len(keys) > n:
        keep = np.sort(rng.choice(len(keys), size=n, replace=False))
        keys = keys[keep]
    return keys.tolist()


def _sample(name, rng, size):
    if name == "uniform":
        return rng.integers(0, 2 ** 50, size=size, dtype=np.int64)
    if name == "normal":
        vals = rng.normal(loc=2.0 ** 49, scale=2.0 ** 45, size=size)
    elif name == "lognormal":
        # Heavy right tail: most keys are small and dense, a few are huge.
        vals = rng.lognormal(mean=0.0, sigma=2.0, size=size) * 1e9
    elif name == "clustered":
        vals = _clustered(rng, size)
    else:
        raise ValueError("unknown distribution: {!r}".format(name))
    return np.clip(vals, 0, MAX_KEY - 1).astype(np.int64)


def _clustered(rng, size, num_clusters=64):
    """Gaussian clusters of very different width and weight.

    Meant to resemble data with strong local structure (e.g. space-filling
    curve encodings of map data) where a single global model fits badly.
    """
    centers = rng.uniform(0, 2.0 ** 48, size=num_clusters)
    widths = rng.exponential(scale=2.0 ** 32, size=num_clusters)
    weights = rng.dirichlet(np.full(num_clusters, 0.5))
    which = rng.choice(num_clusters, size=size, p=weights)
    return centers[which] + rng.normal(size=size) * widths[which]


def load_sosd(path: str, limit: Optional[int] = None) -> List[int]:
    """Load a SOSD binary key file, returning sorted unique keys."""
    with open(path, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        if limit is not None:
            count = min(count, limit)
        data = np.fromfile(f, dtype="<u8", count=count)
    keys = np.unique(data)
    if len(keys) and int(keys[-1]) >= MAX_KEY:
        warnings.warn(
            "keys exceed 2**53; model predictions lose precision "
            "(results stay correct, but error bounds get wider)"
        )
    return keys.tolist()


def write_sosd(path: str, keys: Sequence[int]) -> None:
    """Write ``keys`` in the SOSD binary format."""
    arr = np.asarray(keys, dtype="<u8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(arr)))
        arr.tofile(f)
