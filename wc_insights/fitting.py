"""Model fitting orchestration (`python app.py fit`).

Pulls historical results, fits the Dixon-Coles attack/defense ratings, runs the
walk-forward backtest, fits the calibration layer from the out-of-sample
predictions, and writes the versioned artifacts:

    model_params.json     fitted attack/defense + home_adv + rho
    calibration.json      per-market isotonic recalibrators
    backtest_report.json  validation metrics (DC vs baselines, calibration gain)
"""

import json
import os

import pandas as pd

from . import backtest, calibration, db, dixoncoles as dc, historical
from .names import norm

REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest_report.json")

# How heavily a completed World Cup match counts vs a same-day friendly when
# refitting ratings. >1 lets tournament form move the ratings, but kept modest
# so a couple of results don't swing them wildly (form shrinkage in spirit).
WC_IMPORTANCE = 2.0


def wc_results(db_path: str) -> pd.DataFrame:
    """Completed World Cup matches from the live DB, shaped as training rows."""
    conn = db.connect(db_path)
    rows = conn.execute(
        """SELECT th.team_name h, ta.team_name a, m.home_goals hg, m.away_goals ag,
                  th.host_flag host FROM matches m
           JOIN teams th ON m.home_team_id=th.team_id
           JOIN teams ta ON m.away_team_id=ta.team_id
           WHERE m.status='Final' AND m.home_goals IS NOT NULL""").fetchall()
    conn.close()
    data = [{
        "date": historical.LEAKAGE_CUTOFF, "home": norm(r["h"]), "away": norm(r["a"]),
        "hg": int(r["hg"]), "ag": int(r["ag"]),
        "neutral": not bool(r["host"]),  # host nation playing at home isn't neutral
        "tournament": "FIFA World Cup", "age_days": 0, "importance": WC_IMPORTANCE,
    } for r in rows]
    return pd.DataFrame(data)


def update_ratings(db_path: str = db.DEFAULT_DB_PATH, half_life_days: float = 540.0) -> dict:
    """Refit ONLY the Dixon-Coles ratings, folding in completed WC results.

    Fast (no backtest/calibration refit — those stay valid since they were fit
    out-of-sample on history). Called after each ingest so predictions for
    upcoming fixtures reflect tournament form.
    """
    base = historical.load()
    teams = historical.team_universe(base, min_matches=8)
    wc = wc_results(db_path)
    full = pd.concat([base, wc], ignore_index=True) if not wc.empty else base
    params = dc.fit(full, teams, half_life_days=half_life_days)
    params["n_wc_matches"] = int(len(wc))
    dc.save(params)
    return {"wc_matches_included": int(len(wc)), "teams_rated": len(teams)}


def fit_all(since: str = "2014-01-01", n_folds: int = 8, half_life_days: float = 540.0,
            db_path: str = None) -> dict:
    # 1) historical data (cut off at tournament start — no leakage)
    df = historical.load(since=since, force_download=True)
    teams = historical.team_universe(df, min_matches=8)

    # 2) production ratings: history + any completed WC results (up-weighted)
    wc = wc_results(db_path) if db_path and os.path.exists(db_path) else pd.DataFrame()
    prod_df = pd.concat([df, wc], ignore_index=True) if not wc.empty else df
    params = dc.fit(prod_df, teams, half_life_days=half_life_days)
    params["n_wc_matches"] = int(len(wc))
    dc.save(params)

    # 3) walk-forward validation + out-of-sample pairs (history only — no leakage)
    report, oos = backtest.walk_forward(df, teams, n_folds=n_folds, half_life_days=half_life_days)

    # 4) calibration layer from OOS pairs
    cal = calibration.MarketCalibrators.fit_from_oos(oos)
    cal.save()

    report["calibration"] = cal.metrics
    report["teams_rated"] = len(teams)
    report["history_matches"] = int(len(df))
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    return report
