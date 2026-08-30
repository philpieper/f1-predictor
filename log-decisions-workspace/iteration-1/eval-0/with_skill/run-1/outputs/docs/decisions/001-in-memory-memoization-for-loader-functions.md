# 001. In-memory memoization for FastF1 loader functions

## Context
Model training was hitting FastF1 API rate limits. `data_loader.py` already
calls `fastf1.Cache.enable_cache()`, which persists raw HTTP responses to
disk, but `get_season_schedule`, `load_race_results`, and `load_quali_results`
still re-run their full session-load-and-parse logic (and can still trigger
new requests when the on-disk cache is cold or feature engineering re-derives
overlapping race windows) every time they're called with the same
`year`/`round_number` within a single process.

## Decision
Wrap the three loader functions in an `lru_cache`-based `_memoize` decorator
(unbounded size, since the input space is a few hundred year/round pairs)
that returns a defensive copy of any cached `DataFrame` so callers can't
corrupt the shared cached object. This adds a same-process, in-memory layer
on top of FastF1's existing on-disk HTTP cache.

## Alternatives considered
- A new on-disk cache (e.g. pickling parsed DataFrames per year/round) —
  redundant with `build_raw_dataset`'s existing resumability, which already
  persists finished results to `data/processed/raw_results.parquet` and
  skips rounds already present there. A second disk cache would duplicate
  that persistence for no real benefit.
- Increasing `fastf1.Cache`'s scope/TTL or adding request throttling —
  doesn't address redundant in-process calls, and FastF1's cache layer isn't
  ours to tune for this.
- A manual `dict` cache keyed by `(year, round_number)` — equivalent to
  `lru_cache` but with more code to maintain (no eviction, no `cache_info`).

## Consequences
A failed load (returns `None` after retries) is also memoized, so a
transient rate-limit failure won't be retried again within the same process
run — only a fresh run (or an explicit `cache_clear()`) will re-attempt it.
This is an acceptable trade-off since each call already retries
`MAX_LOAD_RETRIES` times internally before giving up.
