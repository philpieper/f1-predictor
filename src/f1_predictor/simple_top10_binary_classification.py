import lightgbm as lgb
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1_predictor.data_loader import PROCESSED_DIR
from f1_predictor.features import FEATURE_COLUMNS, TARGET_COLUMN, time_series_cv_splits


def _score(y_true: pd.Series, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


def _report_cv(name: str, fold_scores: list[dict]) -> dict:
    scores = pd.DataFrame(fold_scores)
    mean, std = scores.mean(), scores.std()
    summary = {
        "accuracy_mean": mean["accuracy"],
        "accuracy_std": std["accuracy"],
        "roc_auc_mean": mean["roc_auc"],
        "roc_auc_std": std["roc_auc"],
        "log_loss_mean": mean["log_loss"],
        "log_loss_std": std["log_loss"],
        "n_folds": len(scores),
    }
    print(
        f"{name:<20} "
        f"accuracy={summary['accuracy_mean']:.3f}±{summary['accuracy_std']:.3f}  "
        f"roc_auc={summary['roc_auc_mean']:.3f}±{summary['roc_auc_std']:.3f}  "
        f"log_loss={summary['log_loss_mean']:.3f}±{summary['log_loss_std']:.3f}  "
        f"(n={summary['n_folds']} folds)"
    )
    return summary


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

        logreg = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        logreg.fit(X_train, y_train)
        logreg_scores.append(_score(y_test, logreg.predict_proba(X_test)[:, 1]))

        gbm = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            min_child_samples=30,
            num_leaves=7,
            verbosity=-1,
        )
        gbm.fit(X_train, y_train)
        gbm_scores.append(_score(y_test, gbm.predict_proba(X_test)[:, 1]))

    summaries = {
        "LogisticRegression": _report_cv("LogisticRegression", logreg_scores),
        "LightGBM": _report_cv("LightGBM", gbm_scores),
    }

    return logreg, summaries  # logreg fitted on the final (largest) fold's training data


if __name__ == "__main__":
    run_binary_classification()