import lightgbm as lgb
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from f1_predictor.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_table,
    time_based_split,
)

# Class-imbalance handling: only ~half the field finishes in the top 10 each
# race, so this isn't a severe imbalance, but it's still worth handling
# deliberately. We use LightGBM's built-in `class_weight="balanced"` instead
# of manually oversampling the minority class. See
# docs/decisions/2026-08-30-class-imbalance-handling.md for the full
# rationale.
CLASS_WEIGHT = "balanced"


def _season_split(df: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    """Pick train/val/test seasons from whatever data is available.

    Uses the most recent season as the test set and the one before that as
    validation, training on everything earlier. Degrades gracefully when
    fewer than three seasons of data have been collected yet.
    """
    seasons = sorted(df["season"].unique())
    if len(seasons) >= 3:
        return list(seasons[:-2]), [seasons[-2]], [seasons[-1]]
    if len(seasons) == 2:
        return [seasons[0]], [seasons[0]], [seasons[1]]
    return list(seasons), list(seasons), list(seasons)


def run_binary_classification():
    """Classify whether a driver will finish in the top 10 based on qualifying position and other features from data/processed/featrues.parquet."""
    df = build_feature_table()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    train_seasons, val_seasons, test_seasons = _season_split(df)
    train, val, test = time_based_split(df, train_seasons, val_seasons, test_seasons)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight=CLASS_WEIGHT,
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Train seasons: {train_seasons}, val: {val_seasons}, test: {test_seasons}")
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(classification_report(y_test, y_pred, target_names=["not_top10", "top10"]))

    return model


if __name__ == "__main__":
    run_binary_classification()
