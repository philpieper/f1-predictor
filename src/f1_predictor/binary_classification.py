import lightgbm as lgb
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1_predictor.data_loader import PROCESSED_DIR
from f1_predictor.features import FEATURE_COLUMNS, TARGET_COLUMN, time_based_split


def _report(name: str, y_true: pd.Series, y_prob) -> None:
    y_pred = (y_prob >= 0.5).astype(int)
    print(
        f"{name:<20} accuracy={accuracy_score(y_true, y_pred):.3f}  "
        f"roc_auc={roc_auc_score(y_true, y_prob):.3f}  "
        f"log_loss={log_loss(y_true, y_prob):.3f}"
    )


def run_binary_classification():
    """Classify whether a driver will finish in the top 10 based on features from data/processed/features.parquet.

    Only ~740 rows across 2 seasons are available, so logistic regression
    (few parameters, strong regularization) is used as the primary model --
    LightGBM is trained alongside it for comparison but is prone to
    overfitting at this sample size. See docs/decisions for details.
    """
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    # Most recent season is the test set; everything before it is training.
    seasons = sorted(df["season"].unique())
    train_seasons, test_seasons = seasons[:-1], seasons[-1:]
    train, _, test = time_based_split(
        df, train_seasons=train_seasons, val_seasons=[], test_seasons=test_seasons
    )

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
    _report("LogisticRegression", y_test, logreg.predict_proba(X_test)[:, 1])

    gbm = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        min_child_samples=30,
        num_leaves=7,
        verbosity=-1,
    )
    gbm.fit(X_train, y_train)
    _report("LightGBM", y_test, gbm.predict_proba(X_test)[:, 1])

    return logreg


if __name__ == "__main__":
    run_binary_classification()