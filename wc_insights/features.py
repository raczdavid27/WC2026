"""Pre-match feature engineering.

Builds the compact, interpretable feature set the spec prioritizes: team
strength (Elo), recent xG form, tournament context, and market signals. The
features are both fed to the baseline model (via MatchInputs) and persisted to
feature_snapshots for auditability.
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from . import db
from .model import MatchInputs, LEAGUE_TEAM_GOALS
from .names import norm
from .value_engine import novig_probs

REF_ELO = 1750.0       # neutral reference for opponent-strength adjustment
OPP_ADJ_K = 0.15       # how much opponent strength reweights recent xG
SHRINK_K = 5.0         # pseudo-matches pulling form toward the league baseline
FORM_CLIP = 0.35       # max |log-lambda| nudge from recent form


def _recent_stats(conn, team_id: str, before_date: str, limit: int = 5) -> List[dict]:
    rows = conn.execute(
        """SELECT * FROM team_match_stats
           WHERE team_id = ? AND match_date < ?
           ORDER BY match_date DESC LIMIT ?""",
        (team_id, before_date, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _avg(rows: List[dict], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _context_for(conn, match_id: str, team_id: str) -> dict:
    row = conn.execute(
        "SELECT * FROM standings_context WHERE match_id=? AND team_id=?",
        (match_id, team_id),
    ).fetchone()
    return dict(row) if row else {}


def market_implied_home_prob(conn, match_id: str) -> Optional[float]:
    """No-vig home-win probability from the latest 1X2 snapshot (market prior)."""
    rows = conn.execute(
        """SELECT selection, decimal_odds, MAX(captured_at) AS ts
           FROM odds_snapshots
           WHERE match_id=? AND market_type='1X2'
           GROUP BY selection""",
        (match_id,),
    ).fetchall()
    by_sel = {r["selection"]: r["decimal_odds"] for r in rows}
    if {"Home", "Draw", "Away"} <= set(by_sel):
        probs = novig_probs([by_sel["Home"], by_sel["Draw"], by_sel["Away"]])
        return probs[0]
    return None


def _form_adjustments(conn, team_id: str, before_date: str) -> Tuple[float, float]:
    """Opponent-adjusted, shrinkage-weighted recent-form nudges (suggestion #5).

    Returns (attack_logadj, defense_logadj) in log-lambda space. A small sample
    is pulled strongly toward the league baseline; xG is reweighted by opponent
    Elo so chances created against strong sides count for more. 0.0 == no signal.
    """
    rows = _recent_stats(conn, team_id, before_date)
    rows = [r for r in rows if r.get("xg_for") is not None]
    n = len(rows)
    if n == 0:
        return 0.0, 0.0

    adj_for, adj_against = [], []
    for r in rows:
        opp_elo = None
        if r.get("opponent_team_id"):
            o = conn.execute("SELECT elo_rating FROM teams WHERE team_id=?",
                             (r["opponent_team_id"],)).fetchone()
            opp_elo = o["elo_rating"] if o else None
        bump = OPP_ADJ_K * ((opp_elo - REF_ELO) / 100.0) if opp_elo else 0.0
        adj_for.append((r["xg_for"] or 0.0) + bump)
        adj_against.append((r.get("xg_against") or 0.0) - bump)

    def shrink(vals):
        mean = sum(vals) / len(vals)
        return (n * mean + SHRINK_K * LEAGUE_TEAM_GOALS) / (n + SHRINK_K)

    def logadj(shrunk):
        return max(-FORM_CLIP, min(FORM_CLIP, math.log(max(0.3, shrunk) / LEAGUE_TEAM_GOALS)))

    return logadj(shrink(adj_for)), logadj(shrink(adj_against))


def _data_freshness_hours(conn, match_id: str) -> float:
    """Hours since the most recent odds snapshot — drives freshness gating."""
    row = conn.execute("SELECT MAX(captured_at) ts FROM odds_snapshots WHERE match_id=?",
                       (match_id,)).fetchone()
    if not row or not row["ts"]:
        return 999.0
    try:
        captured = datetime.fromisoformat(row["ts"])
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - captured).total_seconds() / 3600.0)
    except ValueError:
        return 999.0


def _lineup_info(conn, match_id: str) -> dict:
    row = conn.execute("SELECT * FROM match_lineups WHERE match_id=?", (match_id,)).fetchone()
    return dict(row) if row else {}


def build_inputs(conn, match: dict) -> MatchInputs:
    """Assemble MatchInputs for one match from teams / stats / context."""
    home = dict(conn.execute("SELECT * FROM teams WHERE team_id=?", (match["home_team_id"],)).fetchone())
    away = dict(conn.execute("SELECT * FROM teams WHERE team_id=?", (match["away_team_id"],)).fetchone())
    kickoff = match["kickoff_utc"]

    h_recent = _recent_stats(conn, home["team_id"], kickoff)
    a_recent = _recent_stats(conn, away["team_id"], kickoff)
    h_ctx = _context_for(conn, match["match_id"], home["team_id"])
    a_ctx = _context_for(conn, match["match_id"], away["team_id"])

    missing = sum(1 for v in (_avg(h_recent, "xg_for"), _avg(a_recent, "xg_for")) if v is None)
    missing_ratio = missing / 2.0

    h_att_adj, h_def_adj = _form_adjustments(conn, home["team_id"], kickoff)
    a_att_adj, a_def_adj = _form_adjustments(conn, away["team_id"], kickoff)
    is_host_home = bool(home["host_flag"])
    lineup = _lineup_info(conn, match["match_id"])

    return MatchInputs(
        elo_home=home["elo_rating"],
        elo_away=away["elo_rating"],
        home_field=is_host_home,
        home_name=norm(home["team_name"]),
        away_name=norm(away["team_name"]),
        neutral=not is_host_home,  # WC matches are neutral unless a host plays
        form_xg_for_home=_avg(h_recent, "xg_for"),
        form_xg_against_home=_avg(h_recent, "xg_against"),
        form_xg_for_away=_avg(a_recent, "xg_for"),
        form_xg_against_away=_avg(a_recent, "xg_against"),
        form_att_adj_home=h_att_adj,
        form_def_adj_home=h_def_adj,
        form_att_adj_away=a_att_adj,
        form_def_adj_away=a_def_adj,
        rest_days_home=match.get("rest_days_home"),
        rest_days_away=match.get("rest_days_away"),
        travel_km_home=match.get("travel_km_home"),
        travel_km_away=match.get("travel_km_away"),
        must_win_home=bool(h_ctx.get("must_win_flag")),
        must_win_away=bool(a_ctx.get("must_win_flag")),
        injury_count_home=lineup.get("injury_count_home", 0) or 0,
        injury_count_away=lineup.get("injury_count_away", 0) or 0,
        lineup_confirmed=bool(lineup.get("lineup_confirmed")),
        data_freshness_hours=_data_freshness_hours(conn, match["match_id"]),
        missing_data_ratio=missing_ratio,
    )


def feature_map(conn, match: dict, inp: MatchInputs) -> Dict[str, dict]:
    """Flat, persistable feature dict: name -> {value, group}."""
    market_home = market_implied_home_prob(conn, match["match_id"])
    feats = {
        "elo_diff": {"value": inp.elo_home - inp.elo_away, "group": "strength"},
        "elo_home": {"value": inp.elo_home, "group": "strength"},
        "elo_away": {"value": inp.elo_away, "group": "strength"},
        "form_xg_for_home": {"value": inp.form_xg_for_home, "group": "form"},
        "form_xg_for_away": {"value": inp.form_xg_for_away, "group": "form"},
        "form_xg_against_home": {"value": inp.form_xg_against_home, "group": "form"},
        "form_xg_against_away": {"value": inp.form_xg_against_away, "group": "form"},
        "rest_diff": {
            "value": (inp.rest_days_home - inp.rest_days_away)
            if None not in (inp.rest_days_home, inp.rest_days_away) else None,
            "group": "context",
        },
        "must_win_home": {"value": 1.0 if inp.must_win_home else 0.0, "group": "context"},
        "must_win_away": {"value": 1.0 if inp.must_win_away else 0.0, "group": "context"},
        "market_home_prob_novig": {"value": market_home, "group": "market"},
        "missing_data_ratio": {"value": inp.missing_data_ratio, "group": "uncertainty"},
        "data_freshness_hours": {"value": inp.data_freshness_hours, "group": "uncertainty"},
    }
    return feats


def persist_features(conn, match_id: str, model_version: str, feats: Dict[str, dict]) -> None:
    conn.execute(
        "DELETE FROM feature_snapshots WHERE match_id=? AND model_version=?",
        (match_id, model_version),
    )
    now = db.utc_now()
    for name, meta in feats.items():
        conn.execute(
            """INSERT INTO feature_snapshots
               (feature_snapshot_id, match_id, model_version, feature_name,
                feature_value, feature_generated_at, feature_group)
               VALUES (?,?,?,?,?,?,?)""",
            (db.new_id(), match_id, model_version, name, meta["value"], now, meta["group"]),
        )
