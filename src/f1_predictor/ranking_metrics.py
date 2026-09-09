"""Evaluation metrics for predicting *exact finishing positions*, not top-10 flags.

Why a separate module from `modeling.score_predictions`: accuracy/ROC-AUC/
log-loss score each driver-row independently, which is the wrong frame for a
finishing order. A race assigns each position to exactly one driver, so the
unit of evaluation here is the **race**, not the row -- every function below
scores one race weekend at a time and the reported number is the mean across
races.

Two conventions run through the whole module, and both matter for
interpreting the numbers:

1. **Truth is `finish_position`, over all starters.** It is a clean
   permutation 1..N in all 104 races on record, retirements included (a
   driver who retires is classified at the back). So "true top 8" needs no
   DNF filtering -- the top 8 are finishers by construction.
2. **Predictions rank all starters, including drivers who go on to retire.**
   At lights out you do not know who will fail to finish, so a model that
   ranks a soon-to-retire driver P2 *should* be charged for pushing everyone
   behind them one place up. Filtering DNFs out of the prediction side would
   quietly hand the model information it cannot have on race morning.

`K = 8` throughout, matching the goal of predicting exact positions for the
top 8 only. Metrics are computed over drivers whose *true* position is <= K
(i.e. "where did the actual top 8 get predicted"), except `set_overlap_at_k`
and `top1_accuracy`, which are noted individually.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

DEFAULT_K = 8

RACE_KEYS = ["season", "round"]


def predicted_rank(scores: pd.Series) -> pd.Series:
    """Turn per-driver scores (higher = better) into integer ranks 1..N.

    Ties break by position in the input, so a rank is always an integer and
    the metrics below can compare it directly against `finish_position`.
    NaN scores sort to the back rather than propagating.
    """
    filled = scores.fillna(-np.inf)
    return (-filled).rank(method="first").astype(int)


def grid_baseline_scores(df: pd.DataFrame) -> pd.Series:
    """The baseline every model has to beat: predict the finishing order is
    the starting order.

    Qualifying already carries most of the signal in F1 -- corr(grid,
    finish) is 0.72 on this dataset -- so grid order is a genuinely strong
    predictor, not a strawman. Any ranker that does not clear the numbers
    printed by this module's __main__ is not earning its complexity.

    `grid_position == 0` encodes a pitlane start (13 rows on record), which
    is the *back* of the field, not pole -- so it maps to a score worse than
    any grid slot rather than the best one.
    """
    grid = df["grid_position"].replace(0, np.nan)
    return -grid


def _relevance(true_pos: np.ndarray, k: int) -> np.ndarray:
    """Graded relevance for NDCG: P1 -> k, P2 -> k-1, ..., Pk -> 1, rest 0."""
    return np.maximum(0.0, k - true_pos + 1.0)


def ndcg_at_k(true_pos: np.ndarray, pred_rank: np.ndarray, k: int = DEFAULT_K) -> float:
    """NDCG@k with exponential gain, so the sharp end of the field dominates.

    Uses gain = 2^rel - 1 over the graded relevance above, which makes
    getting P1 right worth roughly twice getting P2 right, and so on. That
    matches how a finishing-order prediction is actually judged -- missing
    the winner is a bigger error than swapping P7 and P8 -- whereas linear
    gain would treat those as comparable.
    """
    rel = _relevance(true_pos, k)
    gain = np.exp2(rel) - 1.0

    top = pred_rank <= k
    dcg = float(np.sum(gain[top] / np.log2(pred_rank[top] + 1.0)))

    ideal = np.sort(gain)[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(1, len(ideal) + 1) + 1.0)))

    return dcg / idcg if idcg > 0 else 0.0


def score_race(
    true_pos: np.ndarray, pred_rank: np.ndarray, k: int = DEFAULT_K
) -> dict:
    """Score one race. `true_pos` is `finish_position`, `pred_rank` the
    output of `predicted_rank`, both covering every starter in that race.
    """
    true_pos = np.asarray(true_pos, dtype=float)
    pred_rank = np.asarray(pred_rank, dtype=float)

    in_true_top_k = true_pos <= k
    t, p = true_pos[in_true_top_k], pred_rank[in_true_top_k]
    err = np.abs(p - t)

    # Spearman/Kendall need >=2 points and non-constant input; a race where
    # fewer than two of the true top k are present cannot produce one.
    if len(t) >= 2:
        spearman = spearmanr(p, t).statistic
        kendall = kendalltau(p, t).statistic
    else:
        spearman = kendall = np.nan

    return {
        "ndcg_at_k": ndcg_at_k(true_pos, pred_rank, k),
        "exact_hit_rate": float(np.mean(err == 0)),
        "within_one_rate": float(np.mean(err <= 1)),
        "mae_at_k": float(np.mean(err)),
        "spearman_at_k": spearman,
        "kendall_at_k": kendall,
        # Set-level, not order-level: did we identify *who* finishes in the
        # points-paying places, ignoring the order among them? Separating
        # this from the order metrics above is diagnostic -- a model can have
        # good overlap and bad ordering, and the fixes for those differ.
        "set_overlap_at_k": float(np.mean(pred_rank[in_true_top_k] <= k)),
        # Predicted-winner accuracy. Scored over the whole field rather than
        # the true top k, since it asks about one specific driver.
        "top1_accuracy": float(np.any((pred_rank == 1) & (true_pos == 1))),
    }


def score_ranking(
    df: pd.DataFrame, scores: pd.Series, k: int = DEFAULT_K
) -> dict:
    """Score a set of races, averaging each metric over race weekends.

    Averaging per race (not pooling all driver-rows) keeps every race
    weighted equally regardless of how many cars started, and is the only
    way metrics like `top1_accuracy` are even defined.

    `scores` must be aligned to `df`'s index; higher = better.
    """
    work = df[RACE_KEYS + ["finish_position"]].copy()
    work["_score"] = scores
    # Withdrawals (2 rows on record) never took the start: no finishing
    # position to predict and no grid slot to predict it from.
    work = work[work["finish_position"].notna()]

    per_race = []
    for _, race in work.groupby(RACE_KEYS, sort=True):
        per_race.append(
            score_race(
                race["finish_position"].to_numpy(),
                predicted_rank(race["_score"]).to_numpy(),
                k=k,
            )
        )

    frame = pd.DataFrame(per_race)
    summary = frame.mean(numeric_only=True).to_dict()
    summary["n_races"] = len(frame)
    return summary


def sample_position_matrix(
    scores: np.ndarray, k: int = DEFAULT_K, n_samples: int = 5000, seed: int = 0
) -> np.ndarray:
    """Monte-Carlo estimate of P(driver i finishes at position j), j in 1..k.

    Draws finishing orders from the Plackett-Luce model implied by `scores`:
    sample the winner from a softmax over the field, remove them, sample
    again from the remainder, and so on. This is the standard
    sampling-without-replacement view of PL, and it is implemented with the
    Gumbel top-k trick -- adding i.i.d. Gumbel noise to each score and
    sorting yields exactly a PL-distributed permutation, so all `n_samples`
    orders come from one vectorised argsort instead of a k-deep loop.

    Its purpose here is evaluation, not modelling: `position_log_loss` needs
    a probability matrix, and this is what turns any scalar-score model --
    including the grid baseline -- into one, so the metric is exercisable
    before a genuinely probabilistic model exists.
    """
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores), scores, -1e9)

    rng = np.random.default_rng(seed)
    gumbel = rng.gumbel(size=(n_samples, len(scores)))
    orders = np.argsort(-(scores[None, :] + gumbel), axis=1)[:, :k]

    matrix = np.zeros((len(scores), k))
    for position in range(k):
        counts = np.bincount(orders[:, position], minlength=len(scores))
        matrix[:, position] = counts / n_samples
    return matrix


def position_log_loss(
    true_pos: np.ndarray,
    position_matrix: np.ndarray,
    k: int = DEFAULT_K,
    eps: float = 1e-6,
) -> dict:
    """Per-position log-loss on a P(driver, position) matrix for one race.

    For each position 1..k, look up the probability the model assigned to
    the driver who actually finished there and take -log of it. This is the
    metric that rewards a *calibrated* answer over a merely well-ordered
    one: a model that says "P(Verstappen wins) = 0.85" and is right scores
    better than one that only says "Verstappen ranks first", and worse when
    it is wrong.

    Returns the mean over positions plus the per-position breakdown, since
    where in the top 8 the uncertainty sits is usually the interesting part.
    """
    true_pos = np.asarray(true_pos, dtype=float)
    losses = {}
    for position in range(1, k + 1):
        who = np.flatnonzero(true_pos == position)
        if len(who) != 1:
            continue
        prob = position_matrix[who[0], position - 1]
        losses[position] = float(-np.log(max(prob, eps)))

    return {
        "position_log_loss_mean": float(np.mean(list(losses.values()))) if losses else np.nan,
        "per_position": losses,
    }


def score_position_log_loss(
    df: pd.DataFrame,
    scores: pd.Series,
    k: int = DEFAULT_K,
    n_samples: int = 5000,
    seed: int = 0,
) -> dict:
    """`position_log_loss` over a set of races, via PL sampling of `scores`.

    Note this charges the *sampling* model as much as the scores: a scalar
    score has no notion of how spread out the field is, so the PL
    temperature is implicitly fixed by the score scale. Treat the absolute
    value as a relative yardstick between models on the same score scale,
    not as an absolute calibration measure -- that arrives with S2, where
    the PL likelihood is fitted rather than assumed.
    """
    work = df[RACE_KEYS + ["finish_position"]].copy()
    work["_score"] = scores
    work = work[work["finish_position"].notna()]

    means = []
    per_position = {position: [] for position in range(1, k + 1)}
    for _, race in work.groupby(RACE_KEYS, sort=True):
        matrix = sample_position_matrix(
            race["_score"].to_numpy(), k=k, n_samples=n_samples, seed=seed
        )
        result = position_log_loss(race["finish_position"].to_numpy(), matrix, k=k)
        means.append(result["position_log_loss_mean"])
        for position, loss in result["per_position"].items():
            per_position[position].append(loss)

    return {
        "position_log_loss_mean": float(np.nanmean(means)),
        "per_position_mean": {
            position: float(np.mean(losses)) if losses else np.nan
            for position, losses in per_position.items()
        },
    }


METRIC_ORDER = [
    "ndcg_at_k",
    "exact_hit_rate",
    "within_one_rate",
    "mae_at_k",
    "spearman_at_k",
    "kendall_at_k",
    "set_overlap_at_k",
    "top1_accuracy",
]


def report_ranking_cv(name: str, fold_summaries: list[dict], k: int = DEFAULT_K) -> dict:
    """Aggregate per-fold summaries into mean ± std and print one row.

    Mirrors `modeling.report_cv` so ranking models slot into the same
    train/evaluate loop shape the binary classifiers already use -- the only
    difference is which metrics come out the other end.
    """
    frame = pd.DataFrame(fold_summaries)
    mean, std = frame.mean(numeric_only=True), frame.std(numeric_only=True)

    summary = {"n_folds": len(frame), "n_races": int(frame["n_races"].sum())}
    for metric in METRIC_ORDER:
        summary[f"{metric}_mean"] = mean[metric]
        summary[f"{metric}_std"] = std[metric]

    def cell(metric: str, spread: bool = True) -> str:
        value = f"{summary[f'{metric}_mean']:.3f}"
        # A single summary (the all-races row) has no spread to report --
        # pandas gives NaN there, which would print as a bogus "±nan".
        deviation = summary[f"{metric}_std"]
        if spread and pd.notna(deviation):
            value += f"±{deviation:.3f}"
        return value

    print(
        f"{name:<24} "
        f"ndcg@{k}={cell('ndcg_at_k')}  "
        f"exact={cell('exact_hit_rate')}  "
        f"±1={cell('within_one_rate', spread=False)}  "
        f"mae={cell('mae_at_k')}  "
        f"rho={cell('spearman_at_k', spread=False)}  "
        f"overlap={cell('set_overlap_at_k', spread=False)}  "
        f"win={cell('top1_accuracy', spread=False)}  "
        f"({summary['n_folds']} folds, {summary['n_races']} races)"
    )
    return summary


def run_baseline_report(n_splits: int = 5, k: int = DEFAULT_K) -> dict:
    """Print the grid-order baseline, both over all races and per CV fold.

    The per-fold row is the one to compare a model against: it is scored on
    exactly the `time_series_cv_splits` test folds a model will be scored on,
    so the two numbers are directly comparable. The all-races row is there
    because the baseline needs no training, so it *can* be measured over
    everything -- and the gap between the two rows is a useful read on how
    much of a fold's result is luck of the draw.
    """
    from f1_predictor.data_loader import PROCESSED_DIR
    from f1_predictor.features import time_series_cv_splits

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    print(f"Grid-order baseline (predict finishing order = starting order), K={k}\n")

    overall = score_ranking(df, grid_baseline_scores(df), k=k)
    report_ranking_cv("GridOrder (all races)", [overall], k=k)

    fold_summaries = [
        score_ranking(test, grid_baseline_scores(test), k=k)
        for _, test in time_series_cv_splits(df, n_splits=n_splits)
    ]
    per_fold = report_ranking_cv("GridOrder (CV folds)", fold_summaries, k=k)

    log_loss = score_position_log_loss(df, grid_baseline_scores(df), k=k)
    print(
        f"\nPosition log-loss (PL-sampled from grid order, all races): "
        f"{log_loss['position_log_loss_mean']:.3f}"
    )
    for position, value in log_loss["per_position_mean"].items():
        print(f"  P{position}: {value:.3f}")

    return {"overall": overall, "per_fold": per_fold, "position_log_loss": log_loss}


if __name__ == "__main__":
    run_baseline_report()
