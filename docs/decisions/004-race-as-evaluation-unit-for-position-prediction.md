# 004. Score exact-position predictions per race, not per driver-row

## Context
The goal moved from "will this driver finish top 10" (a per-row binary label)
to "what exact position will each of the top 8 finish in". `modeling.score_predictions`
-- accuracy/ROC-AUC/log-loss over independent driver-rows -- cannot express
this: a race assigns each position to exactly one driver, so per-row scoring
happily rewards an output that puts three drivers in P1 and nobody in P7.

## Decision
Added `src/f1_predictor/ranking_metrics.py`, where the unit of evaluation is
one race weekend and the reported number is the mean across races. Models
emit a scalar score per driver; `predicted_rank()` turns each race's scores
into integer ranks 1..N, which are compared against `finish_position`.
Metrics: NDCG@8, exact-hit rate, within-±1 rate, MAE@8, Spearman/Kendall over
the true top 8, top-8 set overlap, predicted-winner accuracy, and per-position
log-loss on a P(driver, position) matrix. Reuses `time_series_cv_splits`
unchanged -- it already moves whole race weekends together, which is exactly
the grouping a ranker needs. `report_ranking_cv()` mirrors `modeling.report_cv`
so ranking models drop into the existing train/evaluate loop shape.

NDCG uses exponential gain (`2^rel - 1` over `rel = K - position + 1`) rather
than linear, so missing the winner costs far more than swapping P7/P8 --
measured on synthetic input, a P1/P2 swap scores 0.879 against 0.9999 for a
P7/P8 swap. Averaging is per race rather than pooled over rows so that every
race weighs equally regardless of how many cars started, and because
`top1_accuracy` is not even definable on a pooled set of rows.

## Alternatives considered
- Per-driver position regression scored with MAE/RMSE — rejected: predicts
  positions that don't form a permutation (no mutual-exclusion constraint),
  and MAE alone hides whether the error is at the sharp end of the field.
- Keep using `modeling.score_predictions` on a one-vs-rest "finishes at
  position k" formulation — rejected: 8 independent binary problems throw
  away the constraint that the 8 answers must be consistent with each other.
- Pool all driver-rows and score once — rejected: weights races by car count
  and makes race-level metrics (winner accuracy, set overlap) meaningless.

## Consequences
Ranking metrics live apart from the binary-classification metrics rather than
extending them; the two families are not comparable and shouldn't be printed
in one table. `sample_position_matrix()` (Plackett-Luce sampling via the
Gumbel top-k trick) exists in this module so that `position_log_loss` is
exercisable from any scalar-score model, including the baseline, before a
genuinely probabilistic model is built -- see 005 for the baseline itself.
