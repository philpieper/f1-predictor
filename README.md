# F1 Predictor

Predicts whether an F1 driver will finish a race in the top 10 (points-paying
positions), using historical race and qualifying data pulled from
[FastF1](https://github.com/theOehrly/Fast-F1).

## How it works

1. **Data loading** ([data_loader.py](src/f1_predictor/data_loader.py)) — pulls
   race + qualifying results per season/round from FastF1, caches raw session
   data in `data/cache/`, and flattens everything into one row per
   driver/race in `data/processed/raw_results.parquet`. Loading is resumable:
   already-fetched rounds are skipped on re-run.
2. **Feature engineering** ([features.py](src/f1_predictor/features.py)) —
   builds leak-safe features (rolling driver/constructor form, same-track
   history, championship standing, using only data available before each
   race) into `data/processed/features.parquet`.
3. **Models** — binary classifiers predicting top-10 finish, trained with
   both `LogisticRegression` and `LightGBM`:
   - [simple_top10_binary_classification.py](src/f1_predictor/simple_top10_binary_classification.py) —
     uses rolling driver/constructor averages plus grid/quali position.
   - [last3_quali_binary_classification.py](src/f1_predictor/last3_quali_binary_classification.py) —
     uses each driver's last 3 raw finish positions plus current qualifying
     position.
   - [weighted_ensemble_binary_classification.py](src/f1_predictor/weighted_ensemble_binary_classification.py) —
     blends three sub-models (same feature groups as last3_quali, plus
     same-track history and championship standing) at fixed weights: 25%
     championship position, 50% last3_quali, 25% same-track history.
   - [trained_weights_weighted_ensemble_binary_classification.py](src/f1_predictor/trained_weights_weighted_ensemble_binary_classification.py) —
     same three sub-models as weighted_ensemble, but the blend weights are
     learned per fold by a LogisticRegression meta-model instead of fixed at
     25/50/25.

See [docs/decisions](docs/decisions) for the reasoning behind choices.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

LightGBM on macOS requires the `libomp` runtime:

```bash
brew install libomp
```

## Usage

```bash
make data_loader                          # fetch + cache raw race/quali results
make features                             # build the feature table
make simple_top10_binary_classification    # train/evaluate the rolling-features model
make last3_quali_binary_classification     # train/evaluate the last-3-races model
make weighted_ensemble_binary_classification # train/evaluate the weighted-ensemble model
make trained_weights_weighted_ensemble_binary_classification # train/evaluate the learned-weights ensemble model
make pipeline                             # run everything above end-to-end and update the table below
```

`make pipeline` ([compare_pipeline.py](src/f1_predictor/compare_pipeline.py)) runs `uv sync`,
pulls any new data, rebuilds features, trains/evaluates all models, and
rewrites the comparison table below in place.

## Model comparison

<!-- MODEL_COMPARISON_START -->
*Last generated 2026-09-06 15:33 UTC by `make pipeline`.*

| Feature set | Model | Accuracy | ROC AUC | Log Loss | Folds |
|---|---|---|---|---|---|
| Rolling form (simple_top10) | LogisticRegression | 0.790 ± 0.032 | 0.867 ± 0.027 | 0.461 ± 0.043 | 5 |
| Rolling form (simple_top10) | LightGBM | 0.784 ± 0.025 | 0.843 ± 0.025 | 0.499 ± 0.043 | 5 |
| Last-3 races + quali (last3_quali) | LogisticRegression | 0.795 ± 0.024 | 0.862 ± 0.029 | 0.467 ± 0.045 | 5 |
| Last-3 races + quali (last3_quali) | LightGBM | 0.798 ± 0.029 | 0.846 ± 0.034 | 0.488 ± 0.059 | 5 |
| Standings + last3_quali + track, weighted 25/50/25 (weighted_ensemble) | LogisticRegression | 0.798 ± 0.029 | 0.878 ± 0.025 | 0.470 ± 0.023 | 5 |
| Standings + last3_quali + track, weighted 25/50/25 (weighted_ensemble) | LightGBM | 0.793 ± 0.031 | 0.868 ± 0.020 | 0.468 ± 0.025 | 5 |
| Standings + last3_quali + track, learned weights (trained_weights_weighted_ensemble) | LogisticRegression | 0.796 ± 0.024 | 0.878 ± 0.025 | 0.467 ± 0.025 | 5 |
| Standings + last3_quali + track, learned weights (trained_weights_weighted_ensemble) | LightGBM | 0.799 ± 0.031 | 0.867 ± 0.020 | 0.476 ± 0.019 | 5 |
<!-- MODEL_COMPARISON_END -->

## Project structure

```
src/f1_predictor/
  data_loader.py                          # FastF1 data pull + caching
  features.py                             # leak-safe feature engineering + splits
  simple_top10_binary_classification.py   # rolling-form model
  last3_quali_binary_classification.py    # last-3-race-lag model
  weighted_ensemble_binary_classification.py # standings/last3_quali/track blend (25/50/25)
  trained_weights_weighted_ensemble_binary_classification.py # same blend, learned weights
data/
  cache/                                  # raw FastF1 session cache, by season
  processed/                              # raw_results.parquet, features.parquet
docs/decisions/                           # architecture decision records
```
