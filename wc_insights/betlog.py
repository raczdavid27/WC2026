"""Persistent bet log — survives the ingest DB reset.

The main database is rebuilt on every ingest, and ESPN drops a match's odds once
it finishes, so a recommendation made pre-kickoff would otherwise vanish before
it could be graded. This separate SQLite file records each actionable (Bet/Lean)
recommendation the first time it's flagged, and grades it once the match is final
— giving an honest, accumulating record of how the recommendations actually did.

Linked to matches by `fifa_match_id` (stable across rebuilds). Results start
accumulating from the first refresh where a match is still upcoming; matches that
were already finished when first ingested are never logged (we never had a
pre-kickoff pick for them).
"""

import os
import sqlite3

from . import calibration, db, value_engine as ve

BETLOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bet_log.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    bet_id          TEXT PRIMARY KEY,
    dedup_key       TEXT UNIQUE,
    fifa_match_id   TEXT,
    home            TEXT,
    away            TEXT,
    kickoff_utc     TEXT,
    stage           TEXT,
    market_type     TEXT,
    selection       TEXT,
    line_value      REAL,
    bookmaker       TEXT,
    offered_odds    REAL,
    model_prob      REAL,
    edge_pct        REAL,
    ev_pct          REAL,
    status          TEXT,        -- Bet | Lean (status when first flagged)
    confidence_band TEXT,
    result_status   TEXT,        -- Pending | Win | Loss | Void
    pnl_units       REAL,
    logged_at       TEXT,
    settled_at      TEXT
);
"""


def connect(path: str = BETLOG_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _grade(market_type, selection, line_value, hg, ag):
    """Win/Loss/Void for a selection given the final score (None if ungradeable)."""
    total = hg + ag
    if market_type == "1X2":
        if selection == "Home":
            return "Win" if hg > ag else "Loss"
        if selection == "Away":
            return "Win" if ag > hg else "Loss"
        if selection == "Draw":
            return "Win" if hg == ag else "Loss"
    if market_type == "O/U" and line_value is not None:
        if total == line_value:
            return "Void"
        if selection.startswith("Over"):
            return "Win" if total > line_value else "Loss"
        if selection.startswith("Under"):
            return "Win" if total < line_value else "Loss"
    if market_type == "BTTS":
        both = hg >= 1 and ag >= 1
        if selection == "Yes":
            return "Win" if both else "Loss"
        if selection == "No":
            return "Win" if not both else "Loss"
    return None


def sync(main_conn, betlog_path: str = BETLOG_PATH) -> dict:
    """Log new pre-kickoff Bet/Lean picks, then grade any that have finished."""
    conn = connect(betlog_path)
    try:
        logged = _log_new(conn, main_conn)
        settled = _settle_pending(conn, main_conn)
        conn.commit()
    finally:
        conn.close()
    return {"newly_logged": logged, "newly_settled": settled}


def _log_new(conn, main_conn) -> int:
    """Record actionable picks for matches that haven't kicked off yet."""
    rows = main_conn.execute(
        """SELECT r.market_type, r.selection, r.line_value, r.bookmaker, r.offered_odds,
                  r.model_prob, r.edge_pct_points, r.expected_value_pct,
                  r.recommendation_status, r.confidence_band,
                  m.fifa_match_id, m.kickoff_utc, m.stage,
                  th.team_name home, ta.team_name away
           FROM recommendations r
           JOIN matches m ON r.match_id = m.match_id
           JOIN teams th ON m.home_team_id = th.team_id
           JOIN teams ta ON m.away_team_id = ta.team_id
           WHERE m.status = 'Scheduled'
             AND r.recommendation_status IN ('Bet', 'Lean')""").fetchall()
    n = 0
    for r in rows:
        dedup = f"{r['fifa_match_id']}|{r['market_type']}|{r['selection']}|{r['line_value']}"
        cur = conn.execute(
            """INSERT OR IGNORE INTO bets
               (bet_id, dedup_key, fifa_match_id, home, away, kickoff_utc, stage,
                market_type, selection, line_value, bookmaker, offered_odds, model_prob,
                edge_pct, ev_pct, status, confidence_band, result_status, pnl_units,
                logged_at, settled_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (db.new_id(), dedup, r["fifa_match_id"], r["home"], r["away"], r["kickoff_utc"],
             r["stage"], r["market_type"], r["selection"], r["line_value"], r["bookmaker"],
             r["offered_odds"], r["model_prob"], r["edge_pct_points"], r["expected_value_pct"],
             r["recommendation_status"], r["confidence_band"], "Pending", None,
             db.utc_now(), None))
        n += cur.rowcount
    return n


def _settle_pending(conn, main_conn) -> int:
    """Grade pending bets whose match is now Final in the main DB."""
    pending = conn.execute("SELECT * FROM bets WHERE result_status = 'Pending'").fetchall()
    n = 0
    for b in pending:
        m = main_conn.execute(
            "SELECT status, home_goals, away_goals FROM matches WHERE fifa_match_id=?",
            (b["fifa_match_id"],)).fetchone()
        if not m or m["status"] != "Final" or m["home_goals"] is None:
            continue
        result = _grade(b["market_type"], b["selection"], b["line_value"],
                        m["home_goals"], m["away_goals"])
        if result is None:
            continue
        pnl = ve.settle_pnl(result, b["offered_odds"], 1.0)
        conn.execute(
            "UPDATE bets SET result_status=?, pnl_units=?, settled_at=? WHERE bet_id=?",
            (result, pnl, db.utc_now(), b["bet_id"]))
        n += 1
    return n


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
def _agg(rows) -> dict:
    decided = [r for r in rows if r["result_status"] in ("Win", "Loss")]
    wins = sum(1 for r in decided if r["result_status"] == "Win")
    losses = sum(1 for r in decided if r["result_status"] == "Loss")
    voids = sum(1 for r in rows if r["result_status"] == "Void")
    pending = sum(1 for r in rows if r["result_status"] == "Pending")
    settled = [r for r in rows if r["result_status"] in ("Win", "Loss", "Void")]
    pnl = sum((r["pnl_units"] or 0.0) for r in settled)
    staked = len([r for r in settled if r["result_status"] != "Void"])
    edges = [r["edge_pct"] for r in rows if r["edge_pct"] is not None]
    return {
        "logged": len(rows), "settled": len(settled), "pending": pending,
        "wins": wins, "losses": losses, "voids": voids,
        "win_rate": (wins / (wins + losses)) if (wins + losses) else 0.0,
        "roi_pct": (pnl / staked * 100.0) if staked else 0.0,
        "profit_units": round(pnl, 2),
        "avg_edge_pct": (sum(edges) / len(edges)) if edges else 0.0,
    }


def stats(betlog_path: str = BETLOG_PATH) -> dict:
    conn = connect(betlog_path)
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM bets").fetchall()]
        by_status, by_market = {}, {}
        for key, bucket in (("status", by_status), ("market_type", by_market)):
            for v in sorted({r[key] for r in rows}):
                bucket[v] = _agg([r for r in rows if r[key] == v])
        recent = [dict(r) for r in conn.execute(
            """SELECT home, away, market_type, selection, offered_odds, result_status,
                      pnl_units, edge_pct, settled_at FROM bets
               WHERE result_status IN ('Win','Loss','Void')
               ORDER BY settled_at DESC LIMIT 25""").fetchall()]
        upcoming = [dict(r) for r in conn.execute(
            """SELECT home, away, market_type, selection, offered_odds, status,
                      edge_pct, kickoff_utc FROM bets
               WHERE result_status='Pending' ORDER BY kickoff_utc LIMIT 25""").fetchall()]
    finally:
        conn.close()
    out = _agg(rows)

    # model reliability over decided picks (predicted prob vs actual win)
    pairs = [(r["model_prob"], 1 if r["result_status"] == "Win" else 0)
             for r in rows if r["result_status"] in ("Win", "Loss") and r["model_prob"] is not None]
    out["brier"] = round(calibration.brier_score(pairs), 4) if pairs else None
    out["log_loss"] = round(calibration.log_loss(pairs), 4) if pairs else None
    out["reliability"] = calibration.reliability_buckets(pairs) if pairs else []

    # cumulative P/L over graded picks (settlement order)
    graded = sorted([r for r in rows if r["result_status"] in ("Win", "Loss", "Void")],
                    key=lambda r: r["settled_at"] or "")
    cum, curve = 0.0, []
    for i, r in enumerate(graded, 1):
        cum += r["pnl_units"] or 0.0
        curve.append({"n": i, "cum": round(cum, 2)})

    out.update({"by_status": by_status, "by_market": by_market,
                "recent": recent, "upcoming": upcoming, "roi_curve": curve})
    return out
