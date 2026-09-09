# 005. Grid-order baseline; predicted rankings include drivers who retire

## Context
Two choices had to be pinned down before any position model could be judged
(see 004 for the metrics themselves): what counts as a floor for "good", and
whether drivers who go on to retire belong in the predicted ordering.

## Decision
**Baseline.** `grid_baseline_scores()` predicts the finishing order is the
starting order, and `make ranking_metrics` prints it as the number every model
must clear. It is not a strawman: `corr(grid_position, finish_position)` is
0.716 on the current 104 races, and the baseline scores NDCG@8 0.846, exact
hits 0.247, within-±1 0.528, MAE@8 2.34, Spearman 0.702, winner 0.624 over
the 5 CV test folds. `grid_position == 0` (a pitlane start, 13 rows) maps to
the *back* of the field, not pole -- taking `-grid_position` naively would
score those as the strongest car in the race.

**DNF convention.** Predictions rank all starters, including drivers who
retire; truth is `finish_position`, which is already a clean permutation
1..N in all 104 races with retirements classified at the back. So a model
that ranks a soon-to-retire driver P2 is charged for pushing everyone behind
them one place up.

## Alternatives considered
- Filter retirements out of both sides before ranking — rejected: it leaks
  race-morning-unknowable information into the prediction, and measurably
  flatters the result. Measured over all 104 races, the grid baseline scores
  0.278 exact hits / MAE@8 1.93 / winner 0.644 under that framing, against
  0.232 / 2.42 / 0.577 when soon-to-retire drivers stay in the ordering --
  so roughly a fifth of the apparent exact-hit skill, and half a position of
  MAE, came purely from knowing in advance who wouldn't finish.
- A "predict the championship-standings order" baseline — passed over as a
  weaker floor (it ignores this weekend's qualifying entirely), though it is
  cheap to add later if a per-circuit comparison is wanted.

## Consequences
Numbers from this module are not comparable to any external F1-prediction
figure computed on finishers only -- they will look worse and are measuring a
harder, honest task. Handling retirements is therefore part of the modelling
problem, not preprocessing: a model that predicts *who* fails to finish gains
real credit under these metrics.
