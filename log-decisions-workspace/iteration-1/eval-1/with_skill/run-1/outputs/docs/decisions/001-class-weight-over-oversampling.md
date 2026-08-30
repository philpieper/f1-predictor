# 001. Use LightGBM class_weight balancing instead of manual oversampling

## Context
`binary_classification.py` trains a LightGBM model to predict whether a
driver finishes in the top 10. The target is imbalanced enough (roughly
half the field on a good day, fewer on high-DNF races) that we needed a
documented way to handle it rather than leaving it to chance per training
run.

## Decision
Use LightGBM's built-in `class_weight="balanced"` (reweights the loss
inversely to class frequency) instead of manually oversampling the
minority class before training.

## Alternatives considered
- Manual oversampling (e.g. duplicating/SMOTE-ing minority-class rows) —
  passed over because the pipeline uses a time-based season split
  (`time_based_split` in `features.py`) with leak-safe rolling features;
  oversampling would need to happen strictly after the split to avoid
  leaking duplicated rows across train/val boundaries, adding complexity
  for a dataset that's already small (a few seasons of races).
- Leaving class weights untouched (no rebalancing) — passed over since it
  biases the model toward the majority class and understates recall on
  top-10 finishes, which is the outcome we actually care about predicting.

## Consequences
Rebalancing is computed fresh from each training fold's own class
distribution, so it stays correct if the imbalance ratio shifts season to
season without any extra data-prep step. If we later want finer control
(e.g. a custom weight ratio instead of "balanced"), `scale_pos_weight` on
the same `LGBMClassifier` is the natural next lever.
