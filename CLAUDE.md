# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Predicts whether an F1 driver finishes a race in the top 10 (points-paying
positions), using historical race and qualifying data pulled from
[FastF1](https://github.com/theOehrly/Fast-F1). Two independent binary
classifiers are trained and compared, not a single shipped model.

## Commands

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). LightGBM on macOS
needs the `libomp` runtime (`brew install libomp`).

```bash
uv sync                                    # install/update deps
make data_loader                           # fetch + cache raw race/quali results (resumable)
make features                              # build the feature table from raw results
make simple_top10_binary_classification    # train/evaluate the rolling-features model
make last3_quali_binary_classification     # train/evaluate the last-3-races model
make compare_pipeline                      # run everything end-to-end and rewrite the README comparison table
```

There is no test suite, linter, or formatter configured. To run a single
model/module directly (equivalent to what `make` does):

```bash
uv run python -m f1_predictor.<module_name>
```

## Architecture

Three-stage pipeline, each stage reading the previous stage's parquet output
from `data/processed/`:

1. **[data_loader.py](src/f1_predictor/data_loader.py)** — pulls Race +
   Qualifying sessions per season/round from FastF1, caches raw session data
   in `data/cache/`, flattens to one row per driver/race in
   `data/processed/raw_results.parquet`. `build_raw_dataset()` is resumable:
   it diffs against existing `(season, round)` keys already in the parquet
   file and only fetches what's missing, so a rate limit or crash mid-run is
   safe to just re-run. It also retries a session load up to
   `MAX_LOAD_RETRIES` times if FastF1 returns a "corrupt" result (every row
   missing a position — a sign of silent rate-limiting, not a real DNF-everyone
   race).
2. **[features.py](src/f1_predictor/features.py)** — builds leak-safe features
   into `data/processed/features.parquet`. The one invariant that matters:
   every feature for a race must only use information available before that
   race's lights out. Rolling driver/constructor stats are computed with
   `.shift(1)` inside a groupby before the rolling window, so a race's own
   result never leaks into its own features. This file also owns three
   train/test splitting strategies (`chronological_pct_split`,
   `time_based_split`, `time_series_cv_splits`); only `time_series_cv_splits`
   (chronological k-fold over whole race weekends) is currently used by the
   models — see [docs/decisions/003](docs/decisions/003-time-series-cv-over-single-split.md)
   for why the other two were superseded but left in place.
3. **Models** — both train `LogisticRegression` (primary/shipped) and
   `LightGBM` (comparison only) on the same CV splits, and report mean ± std
   accuracy/ROC-AUC/log-loss across folds:
   - [simple_top10_binary_classification.py](src/f1_predictor/simple_top10_binary_classification.py)
     uses `FEATURE_COLUMNS` from `features.py` (rolling driver/constructor
     averages + grid/quali position).
   - [last3_quali_binary_classification.py](src/f1_predictor/last3_quali_binary_classification.py)
     defines its own smaller `FEATURE_COLUMNS` (each driver's raw last-3
     finish positions + current quali position) and its own feature-loading
     function (`build_last3_quali_features`), since these are unsmoothed
     per-race lags that require dropping rows without 3 prior races, unlike
     the rolling averages the other model uses.

[compare_pipeline.py](src/f1_predictor/compare_pipeline.py) orchestrates all
of the above end-to-end (`uv sync` → data pull → feature build → both
models) and rewrites the model-comparison table in `README.md` between the
`<!-- MODEL_COMPARISON_START/END -->` markers — those markers must not be
removed or reordered relative to each other.

## Working in this codebase

- When changing feature engineering or splitting logic, check whether the
  reasoning is already captured in [docs/decisions](docs/decisions) before
  re-deriving it — several past changes (chronological split, CV over a
  single split, logistic regression over LightGBM) were driven by measured
  instability at this dataset's small size (~740 rows, ~33-36 race rounds),
  not by a general preference for one algorithm.
- Preserve the leak-safety invariant in `features.py` (shift-before-roll) for
  any new feature — it's the most important correctness property in the repo
  and won't be caught by any test since there is no test suite.
- `SEASONS` is duplicated between `data_loader.py`'s `__main__` block and
  `compare_pipeline.py` — widen both together if adding a season.
