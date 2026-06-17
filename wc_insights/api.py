"""REST response builders for /api/v1/*.

Pure query/shape functions; the HTTP plumbing lives in server.py. Every
response carries model_version / data freshness / timestamp metadata for
auditability, per the spec.
"""

from typing import Optional

from . import db, model, pipeline
from . import MODEL_VERSION


def _meta(conn) -> dict:
    last_pred = conn.execute(
        "SELECT MAX(prediction_generated_at) FROM model_outputs").fetchone()[0]
    meta = {
        "model_version": MODEL_VERSION,
        "generated_at": db.utc_now(),
        "last_prediction_at": last_pred,
    }
    meta.update(model.dc_info())
    return meta


def _team(conn, team_id):
    r = conn.execute("SELECT * FROM teams WHERE team_id=?", (team_id,)).fetchone()
    return dict(r) if r else None


def _top_recommendation(conn, match_id):
    r = conn.execute(
        """SELECT * FROM recommendations WHERE match_id=?
           ORDER BY CASE recommendation_status WHEN 'Bet' THEN 0 WHEN 'Lean' THEN 1
                    ELSE 2 END, edge_pct_points DESC LIMIT 1""",
        (match_id,)).fetchone()
    return dict(r) if r else None


def _primary_book(conn, match_id):
    """The bookmaker with the most 1X2 snapshots for this match (book-agnostic)."""
    row = conn.execute(
        """SELECT bookmaker, COUNT(*) c FROM odds_snapshots
           WHERE match_id=? AND market_type='1X2'
           GROUP BY bookmaker ORDER BY c DESC LIMIT 1""", (match_id,)).fetchone()
    return row["bookmaker"] if row else None


def _line_movement_summary(conn, match_id):
    """Compare opening vs latest no-vig home prob to summarize 1X2 drift."""
    book = _primary_book(conn, match_id)
    if not book:
        return None
    rows = conn.execute(
        """SELECT is_opening, captured_at, selection, decimal_odds FROM odds_snapshots
           WHERE match_id=? AND market_type='1X2' AND bookmaker=?
           ORDER BY captured_at""", (match_id, book)).fetchall()
    if not rows:
        return None
    opening = [r for r in rows if r["is_opening"]]
    latest_ts = max(r["captured_at"] for r in rows)
    latest = [r for r in rows if r["captured_at"] == latest_ts]

    def home_odds(group):
        for r in group:
            if r["selection"] == "Home":
                return r["decimal_odds"]
        return None
    o0, o1 = home_odds(opening), home_odds(latest)
    if not o0 or not o1:
        return None
    delta = (1 / o1) - (1 / o0)
    direction = "shortening" if delta > 0.005 else "drifting" if delta < -0.005 else "stable"
    return {"home_implied_open": round(1 / o0, 3), "home_implied_now": round(1 / o1, 3),
            "direction": direction, "delta_pp": round(delta * 100, 2)}


# --------------------------------------------------------------------------
def list_matches(conn, stage=None, date=None, recommendation_status=None, market_type=None):
    q = "SELECT * FROM matches WHERE 1=1"
    params = []
    if stage:
        q += " AND stage=?"; params.append(stage)
    if date:
        q += " AND substr(kickoff_utc,1,10)=?"; params.append(date)
    q += " ORDER BY kickoff_utc"
    matches = [dict(r) for r in conn.execute(q, params).fetchall()]

    out = []
    for m in matches:
        top = _top_recommendation(conn, m["match_id"])
        if recommendation_status and (not top or top["recommendation_status"].lower() != recommendation_status.lower()):
            continue
        if market_type and (not top or top["market_type"] != market_type):
            continue
        out.append({
            "match_id": m["match_id"],
            "kickoff_utc": m["kickoff_utc"],
            "stage": m["stage"],
            "group_name": m["group_name"],
            "status": m["status"],
            "home_team": _team(conn, m["home_team_id"])["team_name"],
            "away_team": _team(conn, m["away_team_id"])["team_name"],
            "venue": m["venue_name"],
            "score": (f"{m['home_goals']}-{m['away_goals']}"
                      if m["home_goals"] is not None else None),
            "top_recommendation": (
                {"market_type": top["market_type"], "selection": top["selection"],
                 "status": top["recommendation_status"], "bookmaker": top["bookmaker"],
                 "offered_odds": top["offered_odds"]} if top else None),
            "confidence_band": top["confidence_band"] if top else None,
            "best_edge_pct": round(top["edge_pct_points"], 2) if top else None,
            "best_ev_pct": round(top["expected_value_pct"], 2) if top else None,
            "line_movement_summary": _line_movement_summary(conn, m["match_id"]),
        })
    return {"meta": _meta(conn), "count": len(out), "matches": out}


def match_detail(conn, match_id):
    m = conn.execute("SELECT * FROM matches WHERE match_id=?", (match_id,)).fetchone()
    if not m:
        return None
    m = dict(m)
    home, away = _team(conn, m["home_team_id"]), _team(conn, m["away_team_id"])
    out = conn.execute(
        "SELECT * FROM model_outputs WHERE match_id=? ORDER BY prediction_generated_at DESC LIMIT 1",
        (match_id,)).fetchone()
    out = dict(out) if out else None

    feats = [dict(r) for r in conn.execute(
        "SELECT feature_name, feature_value, feature_group FROM feature_snapshots WHERE match_id=?",
        (match_id,)).fetchall()]

    # market odds table: best price per market/selection
    odds_rows = [dict(r) for r in conn.execute(
        """SELECT s.market_type, s.selection, s.line_value, s.bookmaker, s.decimal_odds
           FROM odds_snapshots s
           JOIN (SELECT market_type, selection, MAX(captured_at) ts FROM odds_snapshots
                 WHERE match_id=? GROUP BY market_type, selection) l
             ON s.market_type=l.market_type AND s.selection=l.selection AND s.captured_at=l.ts
           WHERE s.match_id=?""", (match_id, match_id)).fetchall()]
    # keep best price per (market, selection)
    best = {}
    for r in odds_rows:
        key = (r["market_type"], r["selection"])
        if key not in best or r["decimal_odds"] > best[key]["decimal_odds"]:
            best[key] = r

    recs = [dict(r) for r in conn.execute(
        "SELECT * FROM recommendations WHERE match_id=? ORDER BY edge_pct_points DESC",
        (match_id,)).fetchall()]

    line_points = _line_chart_points(conn, match_id)
    ctx = [dict(r) for r in conn.execute(
        "SELECT * FROM standings_context WHERE match_id=?", (match_id,)).fetchall()]

    return {
        "meta": _meta(conn),
        "match": {
            "match_id": m["match_id"], "stage": m["stage"], "group_name": m["group_name"],
            "kickoff_utc": m["kickoff_utc"], "venue": m["venue_name"], "city": m["city"],
            "weather": m["weather_summary"], "referee": m["referee_name"], "status": m["status"],
            "score": (f"{m['home_goals']}-{m['away_goals']}" if m["home_goals"] is not None else None),
            "rest_days_home": m["rest_days_home"], "rest_days_away": m["rest_days_away"],
        },
        "home_team": home, "away_team": away,
        "strength_comparison": {
            "elo_home": home["elo_rating"], "elo_away": away["elo_rating"],
            "elo_diff": round(home["elo_rating"] - away["elo_rating"], 1),
            "rank_home": home["fifa_rank"], "rank_away": away["fifa_rank"],
        },
        "model": out,
        "fair_vs_market": list(best.values()),
        "recommendations": recs,
        "features": feats,
        "context": ctx,
        "risk_flags": _risk_flags(out, m, dict(conn.execute(
            "SELECT * FROM match_lineups WHERE match_id=?", (match_id,)).fetchone() or {})),
        "line_chart_points": line_points,
        "result": _match_result(conn, m),
    }


def _match_result(conn, m):
    """Full-time result: goal scorers + per-team match statistics (Final only)."""
    if m["status"] != "Final" or m["home_goals"] is None:
        return None
    home_id, away_id = m["home_team_id"], m["away_team_id"]

    def side(team_id):
        return "home" if team_id == home_id else "away"

    scorers = []
    for r in conn.execute(
        """SELECT player_name, minute, event_type, team_id FROM match_events
           WHERE match_id=? AND event_type IN ('goal','penalty','own_goal')
           ORDER BY sort_order""", (m["match_id"],)).fetchall():
        scorers.append({"player": r["player_name"], "minute": r["minute"],
                        "type": r["event_type"], "side": side(r["team_id"])})

    cards = []
    for r in conn.execute(
        """SELECT player_name, minute, event_type, team_id FROM match_events
           WHERE match_id=? AND event_type IN ('yellow','red')
           ORDER BY sort_order""", (m["match_id"],)).fetchall():
        cards.append({"player": r["player_name"], "minute": r["minute"],
                      "type": r["event_type"], "side": side(r["team_id"])})

    stats = {}
    for r in conn.execute("SELECT * FROM team_match_stats WHERE match_id=?", (m["match_id"],)).fetchall():
        r = dict(r)
        stats[side(r["team_id"])] = {
            "goals": r["goals_for"], "xg": r["xg_for"], "shots": r["shots"],
            "shots_on_target": r["shots_on_target"], "corners": r["corners"],
            "possession_pct": r["possession_pct"], "fouls": r["fouls"],
            "offsides": r["offsides"], "yellow": r["cards_yellow"], "red": r["cards_red"],
            "saves": r["saves"],
        }
    return {
        "home_goals": m["home_goals"], "away_goals": m["away_goals"],
        "scorers": scorers, "cards": cards,
        "stats": stats if stats else None,
    }


def _risk_flags(out, m, lineup):
    flags = []
    if out and out["confidence_score"] is not None and out["confidence_score"] < 0.55:
        flags.append("Low model confidence")
    if m["status"] == "Scheduled" and not (lineup and lineup.get("lineup_confirmed")):
        flags.append("Lineups not yet confirmed (pre-match estimate)")
    inj = (lineup or {}).get("injury_count_home", 0) + (lineup or {}).get("injury_count_away", 0) \
        if lineup else 0
    if inj >= 2:
        flags.append(f"{inj} key absentee(s) reported")
    if out and _data_stale(out):
        flags.append("Odds data may be stale")
    return flags


def _data_stale(out):
    return False  # placeholder hook; freshness flows through confidence_score


def _line_chart_points(conn, match_id):
    rows = [dict(r) for r in conn.execute(
        """SELECT bookmaker, selection, captured_at, decimal_odds
           FROM odds_snapshots
           WHERE match_id=? AND market_type='1X2'
           ORDER BY captured_at""", (match_id,)).fetchall()]
    series = {}
    for r in rows:
        key = f"{r['bookmaker']} {r['selection']}"
        series.setdefault(key, []).append(
            {"t": r["captured_at"], "odds": r["decimal_odds"], "implied": round(1 / r["decimal_odds"], 4)})
    return series


def list_recommendations(conn, status=None, min_edge=None, min_ev=None, bookmaker=None, market_type=None):
    q = """SELECT r.*, m.kickoff_utc, m.stage,
                  th.team_name AS home, ta.team_name AS away
           FROM recommendations r
           JOIN matches m ON r.match_id=m.match_id
           JOIN teams th ON m.home_team_id=th.team_id
           JOIN teams ta ON m.away_team_id=ta.team_id WHERE 1=1"""
    params = []
    if status:
        q += " AND lower(r.recommendation_status)=?"; params.append(status.lower())
    if min_edge is not None:
        q += " AND r.edge_pct_points>=?"; params.append(float(min_edge))
    if min_ev is not None:
        q += " AND r.expected_value_pct>=?"; params.append(float(min_ev))
    if bookmaker:
        q += " AND r.bookmaker=?"; params.append(bookmaker)
    if market_type:
        q += " AND r.market_type=?"; params.append(market_type)
    q += " ORDER BY CASE r.recommendation_status WHEN 'Bet' THEN 0 WHEN 'Lean' THEN 1 ELSE 2 END, r.edge_pct_points DESC"
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    return {"meta": _meta(conn), "count": len(rows), "recommendations": rows}


def performance(conn):
    summary = pipeline.performance_summary(conn)
    summary["meta"] = _meta(conn)
    return summary


def recommendation_stats(conn):
    from . import betlog
    out = betlog.stats()
    out["meta"] = _meta(conn)
    return out
