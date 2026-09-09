"""Shared model-fitting and CV-reporting helpers used by every classifier
module (simple_top10, last3_quali, weighted_ensemble). Kept here instead of
duplicated per module so scoring/reporting logic and the LogisticRegression
and LightGBM hyperparameters stay in one place.
"""

import lightgbm as lgb
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_logreg(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def fit_gbm(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        min_child_samples=30,
        num_leaves=7,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


def score_predictions(y_true: pd.Series, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


def report_cv(name: str, fold_scores: list[dict]) -> dict:
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
