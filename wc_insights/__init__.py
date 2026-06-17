"""FIFA World Cup Betting Insights — pure-Python implementation.

A decision-support app that estimates fair probabilities, compares them
against bookmaker odds, flags value bets, and tracks calibration / CLV.

Modules:
    db          SQLite schema + connection helpers.
    value_engine  No-vig, implied prob, edge, EV, fair odds, CLV (with tests).
    model       Poisson / Elo expected-goals baseline model.
    features    Pre-match feature engineering + snapshots.
    recommender Recommendation rules engine + staking.
    calibration Brier / log-loss / reliability buckets.
    seed        Synthetic World Cup fixtures + odds snapshots (demo mode).
    pipeline    Orchestrates feature gen -> scoring -> recommendations.
    api         REST handlers for /api/v1/*.
    server      stdlib http.server wiring + static frontend.
"""

__version__ = "1.1.0"
MODEL_VERSION = "dixon-coles-1.1.0"
