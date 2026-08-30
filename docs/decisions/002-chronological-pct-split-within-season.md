# 002. Split train/test chronologically within each season, not by whole season

## Context
With only 2 seasons of data (2025, 2026), the original split held out all of
the most recent season (2026) as test. That meant training never saw any
signal from the current season -- new car regs, driver lineup changes -- so
the model was blind to whatever shifted between seasons it was being tested on.

## Decision
Added `chronological_pct_split()` in `src/f1_predictor/features.py`: for each
season, sort rounds and take the first 80% as train, the last 20% as test.
`simple_top10_binary_classification.py` and `last3_quali_binary_classification.py`
now use this instead of the whole-season holdout.

## Alternatives considered
- Keep whole-season holdout (train on 2025, test on 2026) — simple, but training
  never sees any 2026 signal, biasing the model against the season it's tested on.
- Random shuffle split — rejected outright: would leak future races into
  training and invalidate the time-based evaluation entirely.

## Consequences
Test rounds are now late-season rather than a full held-out season, so
reported accuracy/ROC-AUC dropped slightly (e.g. simple_top10 accuracy
0.803 -> 0.778). This is expected: less same-season signal to lean on, and
considered a more trustworthy estimate rather than a regression. The old
whole-season `time_based_split()` is left in `features.py`, unused by these
two scripts.
