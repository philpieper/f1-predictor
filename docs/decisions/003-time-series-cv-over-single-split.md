# 003. Evaluate with chronological cross-validation, not a single split

## Context
Supersedes 002. With ~33-36 total race rounds, a single `chronological_pct_split`
cutoff puts only a handful of rounds in the test set. Moving `train_frac` from
0.8 to 0.9 alone swung LogisticRegression accuracy from 0.789 to 0.855 on the
last3_quali script -- almost entirely test-set variance (fewer, differently-composed
rounds), not a real model improvement.

## Decision
Added `time_series_cv_splits()` in `src/f1_predictor/features.py`, wrapping
sklearn's `TimeSeriesSplit` over the sorted list of unique (season, round) race
weekends (so a fold never splits one race weekend across train/test). Both
`simple_top10_binary_classification.py` and `last3_quali_binary_classification.py`
now loop over 5 chronological folds and report mean ± std per metric instead of
a single train/test number.

## Alternatives considered
- Keep tuning `chronological_pct_split`'s `train_frac` — rejected: no single
  cutoff is representative with this few rounds; the metric is too noisy to
  compare configurations against.
- Plain (shuffled) k-fold cross-validation — rejected: folds would ignore
  chronology, letting a test fold sit adjacent in time to near-identical
  train rows (same driver hot streak, same car upgrade window), inflating
  apparent performance and not matching how the model is actually used
  (predicting forward only).

## Consequences
Metrics are now a mean ± std across 5 folds, which is a steadier basis for
comparing feature sets or configurations than any single split. `chronological_pct_split`
is left in `features.py`, unused by these two scripts.
