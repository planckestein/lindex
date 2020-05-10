# Changelog

## 0.2.0 - 2020-05-10

- RadixSpline: GreedySplineCorridor spline plus a radix table on the key prefix.
- PLA index (shrinking-cone segmentation, FITing-Tree style), indexed
  recursively like the PGM-index.
- `bounded_upper_bound` helper for routing through inner levels.

## 0.1.0 - 2020-04-05

- `Index` interface with `lower_bound` semantics.
- Baselines: binary search and a static bulk-loaded B+-tree.
- Two-stage RMI with linear models; last-mile search limited to the leaf's
  key range.
- Synthetic distributions (uniform, normal, lognormal, clustered) and a
  loader for SOSD binary files.
- Benchmark harness (`scripts/run_benchmarks.py`).
