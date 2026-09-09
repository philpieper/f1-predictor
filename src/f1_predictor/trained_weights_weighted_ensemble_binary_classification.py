"""Variant of weighted_ensemble_binary_classification.py that learns the
blend weights instead of using the fixed GROUP_WEIGHTS (25/50/25).

A coarse 0.1-step grid search over the weight simplex found the fixed
weights already near-optimal (best grid gain was +0.0016 ROC-AUC for
LogisticRegression, +0.0002 for LightGBM -- both far inside the +-0.02 CV
noise), which argues against hand-tuning them further. This module checks
the other direction: what does the data actually learn as blend weights,
if given the chance?

Each group's sub-model is fit on an inner "base" chronological slice of the
training fold; its predictions on a held-out inner "meta" slice (never seen
by the sub-models) are then used to fit a small LogisticRegression meta-model
whose coefficients are the learned blend weights. This keeps the meta-model
honest -- it never sees predictions from data the sub-models were trained on.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from f1_predictor.features import chronological_pct_split, time_series_cv_splits
from f1_predictor.modeling import fit_gbm, fit_logreg, report_cv, score_predictions
from f1_predictor.weighted_ensemble_binary_classification import (
    CHAMPIONSHIP_FEATURES,
    LAST3_QUALI_FEATURES,
    TARGET_COLUMN,
    TRACK_FEATURES,
    build_weighted_ensemble_features,
)

GROUPS = {
    "championship": CHAMPIONSHIP_FEATURES,
    "last3_quali": LAST3_QUALI_FEATURES,
    "track_history": TRACK_FEATURES,
}

# Fraction of each outer training fold held out (chronologically, after the
# base slice) to fit the meta-model on sub-model predictions it never saw
# during sub-model training.
META_FRAC = 0.3


def _fit_group_submodels_and_meta(fit_fn, train: pd.DataFrame):
    """Fit one sub-model per group on a base slice, then a LogisticRegression
    meta-model on those sub-models' predictions over a held-out meta slice.

    Returns (sub_models, meta_model). meta_model.coef_ are the learned blend
    weights (not constrained to sum to 1, unlike GROUP_WEIGHTS).
    """
    base_train, meta_train = chronological_pct_split(train, train_frac=1 - META_FRAC)

    sub_models = {}
    meta_features = {}
    for group_name, cols in GROUPS.items():
        model = fit_fn(base_train[cols], base_train[TARGET_COLUMN])
        sub_models[group_name] = model
        meta_features[group_name] = model.predict_proba(meta_train[cols])[:, 1]

    meta_X = pd.DataFrame(meta_features)
    meta_model = LogisticRegression()
    meta_model.fit(meta_X, meta_train[TARGET_COLUMN])

    return sub_models, meta_model


def _predict_with_trained_weights(sub_models: dict, meta_model, test: pd.DataFrame) -> np.ndarray:
    test_features = pd.DataFrame(
        {
            group_name: model.predict_proba(test[GROUPS[group_name]])[:, 1]
            for group_name, model in sub_models.items()
        }
    )
    return meta_model.predict_proba(test_features)[:, 1]


def run_trained_weights_weighted_ensemble_binary_classification(n_splits: int = 5):
    """Same three feature groups and CV protocol as weighted_ensemble, but
    the blend weights are learned per fold instead of fixed at 25/50/25.

    Prints the learned weights per fold (as the meta-model's coefficients)
    alongside the usual accuracy/ROC-AUC/log-loss so they can be eyeballed
    against GROUP_WEIGHTS.
    """
    df = build_weighted_ensemble_features()
    group_names = list(GROUPS.keys())

    logreg_scores = []
    gbm_scores = []
    for train, test in time_series_cv_splits(df, n_splits=n_splits):
        y_test = test[TARGET_COLUMN]

        logreg_subs, logreg_meta = _fit_group_submodels_and_meta(fit_logreg, train)
        logreg_probs = _predict_with_trained_weights(logreg_subs, logreg_meta, test)
        logreg_scores.append(score_predictions(y_test, logreg_probs))
        print(
            "  LogisticRegression learned weights: "
            + ", ".join(
                f"{name}={w:.2f}"
                for name, w in zip(group_names, logreg_meta.coef_[0])
            )
        )

        gbm_subs, gbm_meta = _fit_group_submodels_and_meta(fit_gbm, train)
        gbm_probs = _predict_with_trained_weights(gbm_subs, gbm_meta, test)
        gbm_scores.append(score_predictions(y_test, gbm_probs))
        print(
            "  LightGBM learned weights:           "
            + ", ".join(
                f"{name}={w:.2f}"
                for name, w in zip(group_names, gbm_meta.coef_[0])
            )
        )

    summaries = {
        "LogisticRegression": report_cv("LogisticRegression", logreg_scores),
        "LightGBM": report_cv("LightGBM", gbm_scores),
    }

    return summaries


if __name__ == "__main__":
    run_trained_weights_weighted_ensemble_binary_classification()
