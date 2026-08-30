# 001. Logistic regression as primary model for top-10 classification

## Context
`run_binary_classification` needed a model to predict whether a driver
finishes top 10. `features.parquet` only has ~740 rows across 2 seasons
(2025 train, 2026 test), with 8 numeric features and a roughly balanced
target (43%/57%).

## Decision
Use `LogisticRegression` (scaled + median-imputed) as the primary model.
`LGBMClassifier` (shallow, heavily regularized) is trained alongside it
purely for comparison, not as the shipped model.

## Alternatives considered
- LightGBM as primary — already a project dependency and typically strong
  on tabular data, but with only ~740 rows / 2 seasons it's prone to
  overfitting relative to its flexibility. Measured: logistic regression
  scored roc_auc=0.873 vs LightGBM's 0.853 on the 2026 test split.
- Random forest — similar overfitting risk to LightGBM at this sample
  size, with less interpretability upside than logistic regression.

## Consequences
Revisit once more seasons of results accumulate (more rows, more distinct
train seasons) — LightGBM likely overtakes logistic regression as sample
size grows. LightGBM's `libomp` runtime dependency on macOS also needed a
one-time `brew install libomp` to run locally.
