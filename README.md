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
```

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
