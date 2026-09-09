import pandas as pd

from f1_predictor.data_loader import PROCESSED_DIR
from f1_predictor.features import time_series_cv_splits
from f1_predictor.modeling import fit_gbm, fit_logreg, report_cv, score_predictions

FEATURE_COLUMNS = [
    "finish_last1",
    "finish_last2",
    "finish_last3",
    "quali_position",
]

TARGET_COLUMN = "target_top10"


def build_last3_quali_features() -> pd.DataFrame:
    """One row per driver/race: last 3 individual finish positions + this
    race's own qualifying position, to predict this race's top-10 finish.

    Example: NOR finished P1, P3, P4 in the last three races and qualifies
    P2 for race 4 -> predict whether NOR finishes top 10 in race 4.

    Unlike the rolling averages in features.py, finish_last1-3 are raw
    per-race lags (not smoothed), so a driver needs at least 3 prior races
    on record before a row is usable -- rows without that history are
    dropped.
    """
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df = df.dropna(subset=["finish_last1", "finish_last2", "finish_last3", "quali_position"])

    return df.sort_values(["season", "round", "driver"]).reset_index(drop=True)


def run_last3_quali_binary_classification(n_splits: int = 5):
    """Classify top-10 finish from a driver's last 3 race results + current quali position.

    Evaluated with chronological (TimeSeriesSplit) cross-validation rather
    than a single train/test cutoff: with only ~33 race rounds total, one
    cutoff can land on an easy or hard stretch of races and swing the
    reported metrics by a lot. Averaging across folds gives a steadier
    estimate. See docs/decisions/003 for details.
    """
    df = build_last3_quali_features()

    logreg_scores = []
    gbm_scores = []
    logreg = None
    for train, test in time_series_cv_splits(df, n_splits=n_splits):
        X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
        X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

        logreg = fit_logreg(X_train, y_train)
        logreg_scores.append(score_predictions(y_test, logreg.predict_proba(X_test)[:, 1]))

        gbm = fit_gbm(X_train, y_train)
        gbm_scores.append(score_predictions(y_test, gbm.predict_proba(X_test)[:, 1]))

    summaries = {
        "LogisticRegression": report_cv("LogisticRegression", logreg_scores),
        "LightGBM": report_cv("LightGBM", gbm_scores),
    }

    return logreg, summaries  # logreg fitted on the final (largest) fold's training data


if __name__ == "__main__":
    run_last3_quali_binary_classification()
