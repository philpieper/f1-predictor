"""Pull race weekend data from FastF1 and cache it as flat parquet files.

FastF1 gives us, per session: results (grid, finish position, status),
lap times, and stint/tyre info. We pull Race + Qualifying (+ FP2 for
long-run pace later) for a range of seasons and flatten into one row
per driver per race.
"""

import time
from functools import lru_cache
from pathlib import Path

import fastf1
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SCHEDULE_CACHE_DIR = CACHE_DIR / "schedules"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SCHEDULE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# FastF1's built-in cache (below) covers session data -- results, laps,
# timing -- fetched via fastf1.get_session(...).load(). It does NOT cover
# fastf1.get_event_schedule(), which hits the Ergast/Jolpica API directly
# and isn't backed by FastF1's on-disk HTTP cache. During model training we
# call get_season_schedule() once per season, but that's still one avoidable
# network round-trip per season per run (and per retry), which adds up
# against the rate limit. We add our own on-disk cache for schedules here,
# plus an in-memory layer so repeat calls within one process are free.
fastf1.Cache.enable_cache(str(CACHE_DIR))

# When FastF1's status/timing enrichment gets rate-limited mid-request, it
# doesn't always raise -- it can hand back a results table with blank
# Status and NaN grid/finish positions for every driver. Silently accepting
# that means "corrupt load" and "driver genuinely didn't start" look the
# same. We detect this and retry instead of saving garbage rows.
MAX_LOAD_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def _looks_corrupt(results: pd.DataFrame, position_col: str) -> bool:
    """True if every row is missing position data -- a sign the session
    load silently failed rather than describing a real race outcome.
    """
    positions = pd.to_numeric(results[position_col], errors="coerce")
    return positions.isna().all()


@lru_cache(maxsize=None)
def _get_season_schedule_cached(year: int) -> pd.DataFrame:
    """On-disk + in-memory cached fetch of the season schedule.

    fastf1.get_event_schedule() isn't covered by FastF1's own HTTP cache, so
    without this we'd re-hit the schedule API every time the schedule is
    needed. The on-disk cache survives across process runs (e.g. re-running
    training after a rate limit); lru_cache avoids repeat disk reads and
    re-fetches within a single run.
    """
    schedule_path = SCHEDULE_CACHE_DIR / f"{year}.parquet"
    if schedule_path.exists():
        return pd.read_parquet(schedule_path)

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule.to_parquet(schedule_path, index=False)
    return schedule


def get_season_schedule(year: int, force_refetch: bool = False) -> pd.DataFrame:
    """Race calendar for a season, excluding testing events.

    Cached on disk under data/cache/schedules/<year>.parquet and in memory
    per process. Pass force_refetch=True to bypass both and re-fetch (e.g.
    if a schedule was updated mid-season).
    """
    if force_refetch:
        _get_season_schedule_cached.cache_clear()
        schedule_path = SCHEDULE_CACHE_DIR / f"{year}.parquet"
        if schedule_path.exists():
            schedule_path.unlink()

    return _get_season_schedule_cached(year)


def load_race_results(year: int, round_number: int) -> pd.DataFrame | None:
    """One row per driver for a single race: grid, finish, status, quali time.

    Returns None if the session can't be loaded (e.g. cancelled race) or if
    every attempt comes back looking corrupt (rate-limited enrichment).
    """
    results = None
    for attempt in range(1, MAX_LOAD_RETRIES + 1):
        try:
            race = fastf1.get_session(year, round_number, "R")
            race.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as exc:
            print(f"  [skip] {year} round {round_number} race: {exc}")
            return None

        candidate = race.results.copy()
        if candidate.empty:
            return None

        if not _looks_corrupt(candidate, "Position"):
            results = candidate
            break

        print(
            f"  [retry {attempt}/{MAX_LOAD_RETRIES}] {year} round {round_number} race: "
            "all positions blank, likely rate-limited -- retrying"
        )
        time.sleep(RETRY_DELAY_SECONDS)

    if results is None:
        print(f"  [fail] {year} round {round_number} race: still corrupt after retries")
        return None

    df = pd.DataFrame(
        {
            "season": year,
            "round": round_number,
            "race_name": race.event["EventName"],
            "circuit": race.event["Location"],
            "driver": results["Abbreviation"],
            "driver_id": results["DriverId"] if "DriverId" in results else results["Abbreviation"],
            "constructor": results["TeamName"],
            "grid_position": pd.to_numeric(results["GridPosition"], errors="coerce"),
            "finish_position": pd.to_numeric(results["Position"], errors="coerce"),
            "status": results["Status"],
            "points": results["Points"],
        }
    )

    # DNFs / DSQs have no finish position -> treat separately downstream,
    # don't silently coerce to a fake rank.
    df["classified"] = df["status"].str.contains("Finished|\\+", na=False, regex=True)

    return df


def load_quali_results(year: int, round_number: int) -> pd.DataFrame | None:
    """Best quali time per driver, plus session (Q1/Q2/Q3) reached."""
    results = None
    for attempt in range(1, MAX_LOAD_RETRIES + 1):
        try:
            quali = fastf1.get_session(year, round_number, "Q")
            quali.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as exc:
            print(f"  [skip] {year} round {round_number} quali: {exc}")
            return None

        candidate = quali.results.copy()
        if candidate.empty:
            return None

        if not _looks_corrupt(candidate, "Position"):
            results = candidate
            break

        print(
            f"  [retry {attempt}/{MAX_LOAD_RETRIES}] {year} round {round_number} quali: "
            "all positions blank, likely rate-limited -- retrying"
        )
        time.sleep(RETRY_DELAY_SECONDS)

    if results is None:
        print(f"  [fail] {year} round {round_number} quali: still corrupt after retries")
        return None

    for col in ("Q1", "Q2", "Q3"):
        if col not in results:
            results[col] = pd.NaT

    best_time = results[["Q1", "Q2", "Q3"]].min(axis=1)

    df = pd.DataFrame(
        {
            "season": year,
            "round": round_number,
            "driver": results["Abbreviation"],
            "quali_position": pd.to_numeric(results["Position"], errors="coerce"),
            "quali_best_time_s": best_time.dt.total_seconds(),
        }
    )
    return df


def build_raw_dataset(seasons: list[int], force_refetch: bool = False) -> pd.DataFrame:
    """Pull race + quali results for every round in the given seasons.

    Resumable by default: rounds already present in the existing
    data/processed/raw_results.parquet are skipped rather than re-fetched,
    so re-running after a rate limit or a crash only pulls what's missing.
    Pass force_refetch=True to ignore the existing file and pull everything
    fresh.
    """
    out_path = PROCESSED_DIR / "raw_results.parquet"

    existing = None
    done_keys = set()
    if out_path.exists() and not force_refetch:
        existing = pd.read_parquet(out_path)
        done_keys = set(zip(existing["season"], existing["round"]))

    race_rows = []
    quali_rows = []
    failed_rounds = []

    for year in seasons:
        schedule = get_season_schedule(year, force_refetch=force_refetch)
        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            if round_number == 0:
                continue  # testing marker in some schedules
            if (year, round_number) in done_keys:
                continue
            print(f"Loading {year} round {round_number}: {event['EventName']}")

            race_df = load_race_results(year, round_number)
            quali_df = load_quali_results(year, round_number)

            if race_df is not None:
                race_rows.append(race_df)
            else:
                failed_rounds.append((year, round_number, "race"))

            if quali_df is not None:
                quali_rows.append(quali_df)

    if race_rows:
        races = pd.concat(race_rows, ignore_index=True)
        qualis = pd.concat(quali_rows, ignore_index=True) if quali_rows else pd.DataFrame(
            columns=["season", "round", "driver"]
        )
        new_data = races.merge(qualis, on=["season", "round", "driver"], how="left")
    else:
        new_data = pd.DataFrame()

    if existing is not None and not new_data.empty:
        merged = pd.concat([existing, new_data], ignore_index=True)
    elif existing is not None:
        merged = existing
    else:
        merged = new_data

    merged = merged.sort_values(["season", "round"]).reset_index(drop=True)
    merged.to_parquet(out_path, index=False)
    print(f"Saved {len(merged)} rows to {out_path}")

    if failed_rounds:
        print(f"\n{len(failed_rounds)} round(s) still missing after retries -- re-run to backfill:")
        for year, round_number, kind in failed_rounds:
            print(f"  {year} round {round_number} ({kind})")

    return merged


if __name__ == "__main__":
    # Start small: two recent seasons. Widen once the pipeline works end to end.
    build_raw_dataset(seasons=[2025, 2026])
