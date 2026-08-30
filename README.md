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
   builds leak-safe features (rolling driver/constructor form, using only
   data available before each race) into `data/processed/features.parquet`.
3. **Models** — two binary classifiers predicting top-10 finish, trained with
   both `LogisticRegression` (primary) and `LightGBM` (comparison only):
   - [simple_top10_binary_classification.py](src/f1_predictor/simple_top10_binary_classification.py) —
     uses rolling driver/constructor averages plus grid/quali position.
   - [last3_quali_binary_classification.py](src/f1_predictor/last3_quali_binary_classification.py) —
     uses each driver's last 3 raw finish positions plus current qualifying
     position.

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
make pipeline                             # run everything above end-to-end and update the table below
```

`make pipeline` ([compare_pipeline.py](src/f1_predictor/compare_pipeline.py)) runs `uv sync`,
pulls any new data, rebuilds features, trains/evaluates both models, and
rewrites the comparison table below in place.

## Model comparison

<!-- MODEL_COMPARISON_START -->
*Last generated 2026-08-30 09:26 UTC by `make pipeline`.*

| Feature set | Model | Accuracy | ROC AUC | Log Loss | Folds |
|---|---|---|---|---|---|
| Rolling form (simple_top10) | LogisticRegression | 0.810 ± 0.014 | 0.874 ± 0.012 | 0.449 ± 0.023 | 5 |
| Rolling form (simple_top10) | LightGBM | 0.797 ± 0.011 | 0.857 ± 0.013 | 0.474 ± 0.021 | 5 |
| Last-3 races + quali (last3_quali) | LogisticRegression | 0.807 ± 0.013 | 0.870 ± 0.020 | 0.452 ± 0.031 | 5 |
| Last-3 races + quali (last3_quali) | LightGBM | 0.817 ± 0.021 | 0.856 ± 0.029 | 0.468 ± 0.048 | 5 |
<!-- MODEL_COMPARISON_END -->

## Project structure

```
src/f1_predictor/
  data_loader.py                          # FastF1 data pull + caching
  features.py                             # leak-safe feature engineering + splits
  simple_top10_binary_classification.py   # rolling-form model
  last3_quali_binary_classification.py    # last-3-race-lag model
data/
  cache/                                  # raw FastF1 session cache, by season
  processed/                              # raw_results.parquet, features.parquet
docs/decisions/                           # architecture decision records
```
