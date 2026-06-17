"""Historical international results — training data for the rating models.

Source: martj42/international_results (CC-BY) — every men's international since
1872 with date, teams, score, tournament, and a neutral-venue flag. We cache it
locally and expose a normalized, date-filtered view. All loads are cut off at
the tournament start by default so World Cup outcomes can never leak into the
model that predicts them.
"""

import os
from datetime import date

import pandas as pd

from .names import norm

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "historical_results.csv")

# No tournament outcomes before this date may inform predictions.
LEAKAGE_CUTOFF = "2026-06-11"
# Only fit on reasonably modern football; older eras have different scoring.
DEFAULT_SINCE = "2010-01-01"


def download(force: bool = False) -> str:
    """Cache the results CSV locally. Returns the cache path."""
    if force or not os.path.exists(CACHE_PATH):
        df = pd.read_csv(RESULTS_URL)
        df.to_csv(CACHE_PATH, index=False)
    return CACHE_PATH


def load(since: str = DEFAULT_SINCE, cutoff: str = LEAKAGE_CUTOFF,
         force_download: bool = False) -> pd.DataFrame:
    """Return normalized matches in [since, cutoff).

    Columns: date, home, away (normalized), hg, ag, neutral, tournament, age_days.
    age_days is measured from `cutoff` so time-decay weights make recent matches
    matter most for the rating fit.
    """
    download(force=force_download)
    df = pd.read_csv(CACHE_PATH)
    df = df[(df["date"] >= since) & (df["date"] < cutoff)].copy()
    df = df.dropna(subset=["home_score", "away_score"])
    df["home"] = df["home_team"].map(norm)
    df["away"] = df["away_team"].map(norm)
    df["hg"] = df["home_score"].astype(int)
    df["ag"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    ref = pd.Timestamp(cutoff)
    df["age_days"] = (ref - pd.to_datetime(df["date"])).dt.days
    return df[["date", "home", "away", "hg", "ag", "neutral", "tournament", "age_days"]] \
        .sort_values("date").reset_index(drop=True)


def team_universe(df: pd.DataFrame, min_matches: int = 8) -> list:
    """Teams with enough matches to earn a stable rating."""
    counts = pd.concat([df["home"], df["away"]]).value_counts()
    return sorted(counts[counts >= min_matches].index.tolist())
