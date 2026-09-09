import pandas as pd

from f1_predictor.data_loader import PROCESSED_DIR
from f1_predictor.features import FEATURE_COLUMNS, TARGET_COLUMN, time_series_cv_splits
from f1_predictor.modeling import fit_gbm, fit_logreg, report_cv, score_predictions


def run_binary_classification(n_splits: int = 5):
    """Classify whether a driver will finish in the top 10 based on features from data/processed/features.parquet.

    Only ~740 rows across 2 seasons are available, so logistic regression
    (few parameters, strong regularization) is used as the primary model --
    LightGBM is trained alongside it for comparison but is prone to
    overfitting at this sample size. See docs/decisions for details.

    Evaluated with chronological (TimeSeriesSplit) cross-validation rather
    than a single train/test cutoff: with only ~35 race rounds total, one
    cutoff can land on an easy or hard stretch of races and swing the
    reported metrics by a lot. Averaging across folds gives a steadier
    estimate. See docs/decisions/003 for details.
    """
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

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
    run_binary_classification()