#!/usr/bin/env python3
"""Run the lookup benchmark and write the results to a CSV file."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lindex import bench  # noqa: E402
from lindex.datasets import DISTRIBUTIONS, generate, load_sosd  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=1000000, help="keys per data set")
    parser.add_argument("--queries", type=int, default=100000, help="lookups per run")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DISTRIBUTIONS),
        choices=DISTRIBUTIONS,
        help="synthetic distributions to run",
    )
    parser.add_argument(
        "--sosd",
        nargs="*",
        default=[],
        metavar="PATH",
        help="additional SOSD binary files (e.g. books_200M_uint64)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/results.csv")
    args = parser.parse_args(argv)

    results = []
    for name in args.datasets:
        keys = generate(name, args.keys, seed=args.seed)
        results += bench.run_dataset(name, keys, args.queries, seed=args.seed)
    for path in args.sosd:
        keys = load_sosd(path, limit=args.keys)
        name = os.path.basename(path)
        results += bench.run_dataset(name, keys, args.queries, seed=args.seed)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    bench.write_csv(results, args.out)
    print()
    print(bench.to_markdown(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
