#!/usr/bin/env python3
"""Size vs. search-window trade-off for each structure's main parameter.

Only deterministic quantities are measured (index size and the mean log2
search window), so the sweep is cheap and reproducible on any machine.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lindex import BTree, PLAIndex, RadixSpline, RMI  # noqa: E402
from lindex.bench import make_queries, mean_log2_window  # noqa: E402
from lindex.datasets import DISTRIBUTIONS, generate  # noqa: E402

CONFIGS = (
    [("BTree", "leaf_size", v, lambda v=v: BTree(leaf_size=v)) for v in (16, 64, 256)]
    + [
        ("RMI", "num_leaves", v, lambda v=v: RMI(num_leaves=v))
        for v in (2 ** 8, 2 ** 12, 2 ** 16)
    ]
    + [("PLA", "eps", v, lambda v=v: PLAIndex(eps=v)) for v in (8, 32, 128)]
    + [
        ("RadixSpline", "radix_bits", v, lambda v=v: RadixSpline(eps=32, radix_bits=v))
        for v in (8, 12, 18)
    ]
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=1000000)
    parser.add_argument("--queries", type=int, default=20000)
    parser.add_argument(
        "--datasets", nargs="+", default=list(DISTRIBUTIONS), choices=DISTRIBUTIONS
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/sweep.csv")
    args = parser.parse_args(argv)

    rows = []
    for name in args.datasets:
        keys = generate(name, args.keys, seed=args.seed)
        queries = make_queries(keys, args.queries, seed=args.seed)
        for index_name, param, value, factory in CONFIGS:
            index = factory().build(keys)
            row = {
                "dataset": name,
                "index": index_name,
                "param": param,
                "value": value,
                "size_bytes": index.size_bytes(),
                "log2_window": round(mean_log2_window(index, queries), 3),
            }
            rows.append(row)
            print(
                "{dataset:<10} {index:<12} {param}={value:<8} size {size_bytes:>9} B  "
                "log2 window {log2_window:>6.2f}".format(**row)
            )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
