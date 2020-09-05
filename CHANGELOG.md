# Changelog

## 0.3.0 - 2020-09-05

- Lookups are split into `search_window()` and a shared last-mile search;
  the benchmark reports the mean log2 window size next to latency.
- The benchmark verifies every index against `bisect` (including absent
  keys) before timing it.
- `scripts/sweep.py`: size vs. search window for each structure's main
  parameter.
- Search window plot; design notes in `docs/design.md`; first results in
  the README.

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
