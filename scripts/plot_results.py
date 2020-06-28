#!/usr/bin/env python3
"""Plot benchmark results written by run_benchmarks.py."""

import argparse
import csv
import os
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def label(row):
    return "{} ({})".format(row["index"], row["params"]) if row["params"] else row["index"]


def group(rows):
    """{dataset: {label: row}}, keeping the order of the CSV file."""
    out = OrderedDict()
    for row in rows:
        out.setdefault(row["dataset"], OrderedDict())[label(row)] = row
    return out


def plot_bars(rows, path, column, ylabel, title):
    data = group(rows)
    datasets = list(data)
    labels = list(OrderedDict((label(r), None) for r in rows))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, lab in enumerate(labels):
        xs = [d + (i - (len(labels) - 1) / 2.0) * width for d in range(len(datasets))]
        ys = [float(data[ds][lab][column]) if lab in data[ds] else 0 for ds in datasets]
        ax.bar(xs, ys, width, label=lab)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title("{}, {} keys".format(title, rows[0]["num_keys"]))
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(rows, path):
    """Index size vs. lookup latency, one panel per data set."""
    data = group(rows)
    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 3.6), squeeze=False)
    for ax, (ds, by_label) in zip(axes[0], data.items()):
        for lab, row in by_label.items():
            size = max(int(row["size_bytes"]), 1)
            ax.scatter(size, float(row["lookup_ns"]), label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("index size (bytes)")
        ax.set_title(ds)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("lookup latency (ns)")
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="results file from run_benchmarks.py")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args(argv)

    rows = load(args.csv)
    if not rows:
        print("no rows in {}".format(args.csv), file=sys.stderr)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    plot_bars(
        rows,
        os.path.join(args.out_dir, "lookup.png"),
        "lookup_ns",
        "lookup latency (ns)",
        "Lookup latency",
    )
    if "log2_window" in rows[0]:
        plot_bars(
            rows,
            os.path.join(args.out_dir, "log2_window.png"),
            "log2_window",
            "mean log2(search window)",
            "Last-mile search window",
        )
    plot_tradeoff(rows, os.path.join(args.out_dir, "size_vs_lookup.png"))
    print("wrote plots to {}".format(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
