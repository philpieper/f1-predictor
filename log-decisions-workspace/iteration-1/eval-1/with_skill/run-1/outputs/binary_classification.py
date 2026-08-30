"""Binary classifier: will a driver finish in the top 10 (score points)?

Class imbalance handling: we use LightGBM's built-in `class_weight="balanced"`
rather than manually oversampling the minority class. See
docs/decisions/001-class-weight-over-oversampling.md for the reasoning.
"""

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from f1_predictor.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_table,
    time_based_split,
)


def train_model(train: pd.DataFrame, val: pd.DataFrame) -> lgb.LGBMClassifier:
    """Fit a LightGBM classifier with balanced class weighting.

    `class_weight="balanced"` reweights the loss inversely proportional to
    class frequency (computed from the training fold only), so the model
    doesn't need the minority class physically duplicated in the data.
    """
    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET_COLUMN],
        eval_set=[(val[FEATURE_COLUMNS], val[TARGET_COLUMN])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def evaluate_model(model: lgb.LGBMClassifier, test: pd.DataFrame) -> None:
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["not_top10", "top10"]))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")


def run_binary_classification(
    train_seasons: list[int] | None = None,
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
) -> lgb.LGBMClassifier:
    train_seasons = train_seasons or [2023, 2024]
    val_seasons = val_seasons or [2025]
    test_seasons = test_seasons or [2026]

    df = build_feature_table()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    train, val, test = time_based_split(df, train_seasons, val_seasons, test_seasons)

    model = train_model(train, val)

    if not test.empty:
        evaluate_model(model, test)
    else:
        print("No test-season data available yet; skipping evaluation.")

    return model


if __name__ == "__main__":
    run_binary_classification()
