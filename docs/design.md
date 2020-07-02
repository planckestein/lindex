# Design notes

Working notes on the decisions behind the implementations. They are
written for whoever wants to extend the code (most likely me in a few
months).

## Lookup semantics

All structures index a sorted array of **unique** integer keys and answer
`lower_bound(key)`, the position of the first key `>= key`. This is what
SOSD measures and it makes testing simple: every index must agree with
`bisect.bisect_left` on *any* probe key, not only on keys that are present.

Unique keys are a real restriction. With duplicates, the B+-tree separator
search and the learned models both need extra care (a run of equal keys can
span several pages or segments). The loaders deduplicate, which is also
what most learned-index papers do for their lookup experiments.

## Two-phase lookups

Every lookup is split into

1. `search_window(key) -> (lo, hi)`, which uses the index to bound the
   answer to `lo..hi`, and
2. a last-mile binary search over `keys[lo:hi]`.

Phase 2 is shared code (`base.bounded_lower_bound`). It also checks that the
window was correct by looking at `keys[lo - 1]` and `keys[hi]` and falls
back to a full binary search when it was not. The guarantees below mean the
fallback should never trigger; `tests/test_search_window.py` checks the
window directly so a bug cannot hide behind the fallback.

`log2(hi - lo + 1)` approximates the number of binary search steps in phase
2. In Python this is the most useful number to compare: the interpreter
adds roughly the same overhead per executed line to every structure, which
hides the cache-miss behaviour that decides latency in a C++
implementation.

## Error bounds for keys that are not in the data

PLA and RadixSpline guarantee `|pred(x) - pos(x)| <= eps` for every key `x`
they were built on. A lookup key `q` that is *not* present falls between two
consecutive keys `a < q < b` of the same segment, and its answer is
`pos(b) = pos(a) + 1`. Both models are monotone, so

    pos(a) - eps <= pred(a) <= pred(q) <= pred(b) <= pos(b) + eps

which gives `|pred(q) - pos(b)| <= eps + 1`. The window is therefore
`[pred - eps - 1, pred + eps + 1]`, i.e. at most `2 * eps + 3` keys. If `q` is
larger than the last key of its segment, the answer is the first position
of the next segment. Clamping the prediction to the segment's range handles
that case too.

## RMI

* Root and leaves are linear models fit by least squares. The root is
  monotone, so the leaves partition the key array into contiguous slices,
  and the answer for any key routed to leaf `j` is inside slice `j`
  (including the case where `j` is empty). The search window is the leaf's
  error window intersected with its slice.
* The error window is only guaranteed for training keys. For other keys
  the intersection with the slice is what keeps the search bounded.
* Weak spot: the linear root. On lognormal and clustered data most keys go
  to a few leaves, so the slices (and the error windows) get huge, and
  adding leaves helps little. The RMI paper and SOSD use other root models
  (e.g. cubic or radix) for this reason. Not done yet; see open questions.

## PLA (FITing-Tree segmentation, PGM-style recursion)

* Segmentation is the greedy shrinking cone from FITing-Tree: start a
  segment at its first point and keep the cone of feasible slopes. This is
  not optimal; the PGM-index computes the optimal PLA with a convex hull
  algorithm (O'Rourke 1981) and gets fewer segments for the same `eps`.
  The greedy version is simpler and was good enough for the first
  experiments.
* The first keys of the segments are segmented again with
  `eps_internal` until one segment is left. With `eps >= 1` each segment
  covers at least two points, so the levels at least halve.
* Size is 24 bytes per segment (first key, slope, first position).

## RadixSpline

* GreedySplineCorridor uses only integer arithmetic, so the corridor
  test is exact. Spline points are data points, so a lookup that hits a
  spline point exactly returns its position directly.
* The radix table has `(key_range >> shift) + 2` entries, at most
  `2^radix_bits + 1`. At 18 bits it is about 1 MiB and dominates the size
  of the index: the spline itself needs only around 1,000 points for 1M
  keys at `eps = 32`. Fewer radix bits give a much smaller index with a
  slightly longer search over the spline points.

## Things to keep in mind when reading the numbers

* Keys are kept below `2^53`. Above that, float64 cannot represent every
  key and predictions lose precision. Lookups stay correct because of the
  window check, but the windows get wider. Real 64-bit data sets can
  contain such keys; `load_sosd` warns when it sees one.
* Build times include Python-level loops for the segmentation algorithms
  and numpy for the RMI fits, so build times across structures are not
  comparable with the papers.

## Open questions / next steps

* Cubic or radix root model for the RMI (the CDFShop paper by Marcus et
  al., SIGMOD 2020, explores this space systematically).
* Optimal PLA instead of the greedy cone, to see how many segments it saves
  on our data sets.
* Real data: run on the SOSD `books`, `fb`, `osm_cellids` and `wiki_ts`
  data sets with `scripts/run_benchmarks.py --sosd`.
* Updates. Everything here is static. ALEX (Ding et al., SIGMOD 2020) and
  the dynamic PGM-index are the obvious references.
* A C++ port of the hot path, so that latency numbers mean something.
