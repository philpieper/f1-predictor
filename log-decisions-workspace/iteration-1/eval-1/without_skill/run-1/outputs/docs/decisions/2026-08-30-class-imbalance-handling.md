# Class imbalance handling for the top-10 classifier

- Date: 2026-08-30
- Status: accepted
- Scope: `src/f1_predictor/binary_classification.py`

## Context

`run_binary_classification` predicts `target_top10` (did a driver finish in
the top 10?) from `data/processed/features.parquet`. Only 10 of ~20 grid
slots score points each race, so the positive class is a minority relative
to the full field, though not extremely so (roughly 50/50 in a fully-run
race, worse once DNFs/backmarkers are included). We still need a
deliberate policy for this rather than letting LightGBM's default
(unweighted) loss quietly under-predict the minority class.

Two options were considered:

1. **LightGBM's built-in `class_weight="balanced"`** (equivalently,
   `scale_pos_weight` for binary objectives) — reweights the loss function
   per class based on training-set frequency, no data duplication.
2. **Manual oversampling** of the minority class (random duplication or
   SMOTE-style synthetic samples) before training.

## Decision

Use LightGBM's built-in `class_weight="balanced"`.

## Rationale

- **Time-based split safety.** `time_based_split` in `features.py`
  deliberately splits by season so validation/test always look like the
  future, not a random sample (see its docstring). Oversampling duplicates
  or synthesizes rows *before* that split is applied to the resulting
  training fold; done carelessly it's easy to let duplicated/synthetic
  points leak signal across the split boundary via near-duplicate rows in
  train vs. val/test. `class_weight` reweights the loss instead of
  touching the data, so it can't introduce that class of leakage.
- **No new dependency.** `pyproject.toml` already depends on `lightgbm`;
  manual oversampling would either mean hand-rolled duplication logic (easy
  to get subtly wrong) or adding `imbalanced-learn` for SMOTE, which isn't
  in the project today.
- **Less code to maintain.** `class_weight="balanced"` is a single
  constructor argument on `LGBMClassifier`. Oversampling requires writing,
  testing, and maintaining resampling code, and re-deriving the correct
  ratio any time the feature table changes.
- **Cost is comparable.** Both approaches address the same problem (the
  loss function under-weighting the minority class); class weighting gets
  there without inflating the effective training set size or training time.

## Consequences

- `CLASS_WEIGHT = "balanced"` in `binary_classification.py` is the single
  place this policy is set; if we revisit it (e.g. switch to explicit
  `scale_pos_weight` tuned via the validation set), update it there and
  amend this doc rather than silently changing behavior.
- If the class balance shifts a lot (e.g. once DNFs are modeled separately
  and the top-10 rate among *classified* finishers is much more skewed),
  revisit this decision — `class_weight="balanced"` reweights based on
  whatever training data it sees, so it adapts automatically, but the
  qualitative trade-off above should be re-checked.
