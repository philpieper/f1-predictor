"""Build leak-safe features from raw_results.parquet.

The one rule that matters: every feature for a given race must only use
information available *before that race's lights out*. Rolling stats are
computed with `.shift(1)` inside each driver/constructor group so the
current race's own result never leaks into its own features.
"""

from pathlib import Path

import pandas as pd

from f1_predictor.data_loader import PROCESSED_DIR

ROLLING_WINDOW = 5


def _dnf_rate(classified: pd.Series) -> pd.Series:
    return 1 - classified.astype(float)


def load_raw() -> pd.DataFrame:
    path = PROCESSED_DIR / "raw_results.parquet"
    df = pd.read_parquet(path)
    return df.sort_values(["season", "round", "driver"]).reset_index(drop=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target: did the driver finish in the points (top 10)?"""
    df = df.copy()
    df["target_top10"] = (
        df["classified"] & (df["finish_position"] <= 10)
    ).astype(int)
    return df


def add_rolling_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling driver form, computed *before* each race.

    Sorting by (season, round) and using shift(1) before the rolling window
    means row N's features only see rows 1..N-1 for that driver.
    """
    df = df.copy()
    df = df.sort_values(["driver", "season", "round"])

    g = df.groupby("driver", group_keys=False)

    df["driver_avg_finish_last5"] = g["finish_position"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["driver_dnf_rate_last5"] = g["classified"].apply(
        lambda s: _dnf_rate(s).shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["driver_avg_quali_pos_last5"] = g["quali_position"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["driver_points_last5"] = g["points"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).sum()
    )

    return df.sort_values(["season", "round", "driver"]).reset_index(drop=True)


def add_rolling_constructor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling constructor (team/car) pace, same shift-then-roll pattern.

    Two drivers share a car, so this captures car performance independent
    of the individual driver's rolling stats above.
    """
    df = df.copy()
    df = df.sort_values(["constructor", "season", "round"])

    # Average finish position across BOTH cars for the constructor, per race,
    # so a rolling window over races (not rows) reflects team pace.
    team_race_avg = (
        df.groupby(["constructor", "season", "round"])["finish_position"]
        .mean()
        .reset_index()
        .rename(columns={"finish_position": "team_race_avg_finish"})
        .sort_values(["constructor", "season", "round"])
    )
    team_race_avg["team_avg_finish_last5"] = (
        team_race_avg.groupby("constructor", group_keys=False)["team_race_avg_finish"]
        .apply(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )

    df = df.merge(
        team_race_avg[["constructor", "season", "round", "team_avg_finish_last5"]],
        on=["constructor", "season", "round"],
        how="left",
    )

    return df.sort_values(["season", "round", "driver"]).reset_index(drop=True)


def build_feature_table() -> pd.DataFrame:
    df = load_raw()
    df = add_target(df)
    df = add_rolling_driver_features(df)
    df = add_rolling_constructor_features(df)

    out_path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows, {df.shape[1]} cols to {out_path}")
    return df


FEATURE_COLUMNS = [
    "grid_position",
    "quali_position",
    "quali_best_time_s",
    "driver_avg_finish_last5",
    "driver_dnf_rate_last5",
    "driver_avg_quali_pos_last5",
    "driver_points_last5",
    "team_avg_finish_last5",
]

TARGET_COLUMN = "target_top10"


def time_based_split(
    df: pd.DataFrame,
    train_seasons: list[int],
    val_seasons: list[int],
    test_seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by season, not randomly -- validates on the future, not a random sample."""
    train = df[df["season"].isin(train_seasons)].reset_index(drop=True)
    val = df[df["season"].isin(val_seasons)].reset_index(drop=True)
    test = df[df["season"].isin(test_seasons)].reset_index(drop=True)
    return train, val, test


if __name__ == "__main__":
    build_feature_table()
