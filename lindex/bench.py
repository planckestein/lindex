"""Micro-benchmark harness for lookup latency, build time and index size.

The setup follows SOSD: build an index over a sorted array of unique keys,
then time ``lower_bound`` for a batch of keys sampled from the data set.
All numbers come from CPython, so absolute latencies are dominated by
interpreter overhead; the interesting part is how the structures compare
under the same overhead and how their search windows behave on different
distributions.
"""

import csv
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Sequence

from .base import Index
from .binary_search import BinarySearch
from .btree import BTree
from .rmi import RMI

IndexFactory = Callable[[], Index]


@dataclass
class Result:
    dataset: str
    index: str
    params: str
    num_keys: int
    build_ms: float
    size_bytes: int
    lookup_ns: float


def default_indexes() -> List[IndexFactory]:
    return [
        BinarySearch,
        lambda: BTree(leaf_size=64, fanout=16),
        lambda: RMI(num_leaves=2 ** 10),
        lambda: RMI(num_leaves=2 ** 14),
    ]


def make_queries(keys: Sequence[int], num_queries: int, seed: int = 0) -> List[int]:
    """Sample lookup keys uniformly from the indexed keys."""
    rng = random.Random(seed)
    n = len(keys)
    return [keys[rng.randrange(n)] for _ in range(num_queries)]


def format_params(params: dict) -> str:
    return " ".join("{}={}".format(k, v) for k, v in params.items())


def measure(
    factory: IndexFactory,
    dataset: str,
    keys: List[int],
    queries: List[int],
    repeats: int = 3,
) -> Result:
    index = factory()
    t0 = time.perf_counter()
    index.build(keys)
    build_ms = (time.perf_counter() - t0) * 1e3

    lookup = index.lower_bound
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for q in queries:
            lookup(q)
        best = min(best, time.perf_counter() - t0)

    return Result(
        dataset=dataset,
        index=index.name,
        params=format_params(index.params()),
        num_keys=len(keys),
        build_ms=build_ms,
        size_bytes=index.size_bytes(),
        lookup_ns=best / max(len(queries), 1) * 1e9,
    )


def run_dataset(
    dataset: str,
    keys: List[int],
    num_queries: int,
    factories: Sequence[IndexFactory] = None,
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> List[Result]:
    queries = make_queries(keys, num_queries, seed=seed)
    results = []
    for factory in factories or default_indexes():
        r = measure(factory, dataset, keys, queries)
        log(
            "{:<10} {:<13} {:<28} build {:>9.1f} ms  size {:>10} B  "
            "lookup {:>7.0f} ns".format(
                r.dataset, r.index, r.params, r.build_ms, r.size_bytes, r.lookup_ns
            )
        )
        results.append(r)
    return results


def write_csv(results: Sequence[Result], path: str) -> None:
    fields = list(Result.__dataclass_fields__)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def to_markdown(results: Sequence[Result]) -> str:
    lines = [
        "| dataset | index | params | build (ms) | size (KiB) | lookup (ns) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            "| {} | {} | {} | {:.1f} | {:.1f} | {:.0f} |".format(
                r.dataset,
                r.index,
                r.params or "-",
                r.build_ms,
                r.size_bytes / 1024.0,
                r.lookup_ns,
            )
        )
    return "\n".join(lines)
