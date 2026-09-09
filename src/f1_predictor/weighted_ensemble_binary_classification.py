import numpy as np
import pandas as pd

from f1_predictor.data_loader import PROCESSED_DIR
from f1_predictor.features import time_series_cv_splits
from f1_predictor.modeling import fit_gbm, fit_logreg, report_cv, score_predictions

CHAMPIONSHIP_FEATURES = ["championship_position"]
LAST3_QUALI_FEATURES = ["finish_last1", "finish_last2", "finish_last3", "quali_position"]
TRACK_FEATURES = ["driver_avg_finish_at_circuit"]

FEATURE_COLUMNS = CHAMPIONSHIP_FEATURES + LAST3_QUALI_FEATURES + TRACK_FEATURES
TARGET_COLUMN = "target_top10"

# A single model fit over all features would learn its own weights from the
# data and ignore any split we hand it -- so each group is trained as its
# own sub-model and the three predicted probabilities are blended at these
# fixed weights instead. See docs/decisions for details.
GROUP_WEIGHTS = {
    "championship": 0.25,
    "last3_quali": 0.50,
    "track_history": 0.25,
}


def build_weighted_ensemble_features() -> pd.DataFrame:
    """One row per driver/race: championship standing + last-3 finishes/quali
    + same-track history, to predict this race's top-10 finish.

    Each group needs its own history to be populated -- a driver's first
    race at a circuit has no track history, and the first 3 races of a
    driver's career have no last3_quali lags -- so rows missing any group's
    inputs are dropped, same as last3_quali_binary_classification.
    """
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df = df.dropna(subset=FEATURE_COLUMNS)

    return df.sort_values(["season", "round", "driver"]).reset_index(drop=True)


def _blend_predict_proba(fit_fn, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit one sub-model per feature group, blend P(top10) at GROUP_WEIGHTS."""
    groups = {
        "championship": CHAMPIONSHIP_FEATURES,
        "last3_quali": LAST3_QUALI_FEATURES,
        "track_history": TRACK_FEATURES,
    }
    blended = np.zeros(len(test))
    for group_name, cols in groups.items():
        model = fit_fn(train[cols], train[TARGET_COLUMN])
        blended += GROUP_WEIGHTS[group_name] * model.predict_proba(test[cols])[:, 1]
    return blended


def run_weighted_ensemble_binary_classification(n_splits: int = 5):
    """Classify top-10 finish by blending three sub-models -- championship
    standing, last-3-races + quali, and same-track history -- at fixed
    weights (25/50/25) rather than one model over all features.

    Evaluated with chronological (TimeSeriesSplit) cross-validation, same as
    the other feature sets. See docs/decisions/003 for why.
    """
    df = build_weighted_ensemble_features()

    logreg_scores = []
    gbm_scores = []
    for train, test in time_series_cv_splits(df, n_splits=n_splits):
        y_test = test[TARGET_COLUMN]

        logreg_probs = _blend_predict_proba(fit_logreg, train, test)
        logreg_scores.append(score_predictions(y_test, logreg_probs))

        gbm_probs = _blend_predict_proba(fit_gbm, train, test)
        gbm_scores.append(score_predictions(y_test, gbm_probs))

    summaries = {
        "LogisticRegression": report_cv("LogisticRegression", logreg_scores),
        "LightGBM": report_cv("LightGBM", gbm_scores),
    }

    return summaries


if __name__ == "__main__":
    run_weighted_ensemble_binary_classification()
