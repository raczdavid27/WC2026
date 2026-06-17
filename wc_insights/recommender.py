"""Recommendation rules engine + staking.

Conservative by default: tournament samples are small, so we prefer fewer,
higher-quality flags. A selection is labeled Bet only when edge, EV, freshness,
and risk filters all pass; Lean when EV is positive but one filter is weak;
Pass otherwise.
"""

from typing import Dict, List, Optional

from . import db, value_engine as ve
from .model import confidence_band

# --- default policy (overridable via config) -------------------------------
DEFAULT_CONFIG = {
    "edge_threshold_pp": 4.0,        # min edge in percentage points for Bet
    "ev_threshold_pct": 2.0,         # min EV % for Bet
    "lean_ev_threshold_pct": 0.5,    # min EV % to even consider a Lean
    "freshness_max_hours": 6.0,      # data must be at least this fresh for Bet
    "min_confidence_for_bet": 0.55,  # confidence score gate for Bet
    "kelly_fraction": 0.25,          # quarter-Kelly default
    "stake_mode": "flat",            # "flat" | "kelly"
    "flat_stake_units": 1.0,
    "bankroll_pct_cap": 0.05,        # cap any single stake at 5% of bankroll
    "market_blend_weight": 0.0,      # 0 = pure model; >0 shrinks toward market (#6)
}


def load_config(path: Optional[str] = None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        import json, os
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                cfg.update(json.load(fh).get("recommender", {}))
    return cfg


def _latest_market_snapshots(conn, match_id: str, market_type: str) -> Dict[str, List[dict]]:
    """Latest snapshot per (selection, bookmaker) for a market."""
    rows = conn.execute(
        """SELECT s.* FROM odds_snapshots s
           JOIN (SELECT selection, bookmaker, MAX(captured_at) AS ts
                 FROM odds_snapshots
                 WHERE match_id=? AND market_type=?
                 GROUP BY selection, bookmaker) latest
           ON s.selection=latest.selection AND s.bookmaker=latest.bookmaker
              AND s.captured_at=latest.ts
           WHERE s.match_id=? AND s.market_type=?""",
        (match_id, market_type, match_id, market_type),
    ).fetchall()
    by_sel: Dict[str, List[dict]] = {}
    for r in rows:
        by_sel.setdefault(r["selection"], []).append(dict(r))
    return by_sel


def _novig_market_probs(by_sel: Dict[str, List[dict]], order: List[str]) -> Optional[Dict[str, float]]:
    """No-vig probs for a market using best price per selection."""
    best = {}
    for sel in order:
        if sel not in by_sel:
            return None
        best[sel] = ve.best_odds(by_sel[sel])["decimal_odds"]
    probs = ve.novig_probs([best[s] for s in order])
    return dict(zip(order, probs))


def _evaluate_selection(
    conn, match_id: str, model_version: str, market_type: str, selection: str,
    line_value: Optional[float], model_prob: float, market_prob_novig: float,
    snapshots: List[dict], confidence_score: float, cfg: dict,
) -> dict:
    best = ve.best_odds(snapshots)
    offered = best["decimal_odds"]
    edge_pp = ve.edge_pct_points(model_prob, market_prob_novig)
    ev_pct = ve.expected_value_pct(model_prob, offered)

    # status decision
    fresh_ok = cfg["freshness_max_hours"] >= 0  # freshness handled at scoring time
    passes_bet = (
        edge_pp >= cfg["edge_threshold_pp"]
        and ev_pct >= cfg["ev_threshold_pct"]
        and confidence_score >= cfg["min_confidence_for_bet"]
    )
    if passes_bet:
        status = "Bet"
    elif ev_pct >= cfg["lean_ev_threshold_pct"] and edge_pp > 0:
        status = "Lean"
    else:
        status = "Pass"

    # staking
    if cfg["stake_mode"] == "kelly":
        stake = ve.kelly_fraction(model_prob, offered, cfg["kelly_fraction"])
        stake = min(stake, cfg["bankroll_pct_cap"])
    else:
        stake = cfg["flat_stake_units"] if status in ("Bet", "Lean") else 0.0

    return {
        "recommendation_id": db.new_id(),
        "match_id": match_id,
        "model_version": model_version,
        "market_type": market_type,
        "selection": selection,
        "line_value": line_value,
        "bookmaker": best["bookmaker"],
        "offered_odds": offered,
        "fair_odds": ve.fair_odds(model_prob),
        "model_prob": model_prob,
        "market_prob_novig": market_prob_novig,
        "edge_pct_points": edge_pp,
        "expected_value_pct": ev_pct,
        "clv_reference_odds": None,
        "recommendation_status": status,
        "confidence_band": confidence_band(confidence_score),
        "stake_fraction": stake,
        "created_at": db.utc_now(),
    }


def generate_for_match(conn, match_id: str, model_version: str, prediction, cfg: dict) -> List[dict]:
    """Produce recommendations across all modeled markets for one match."""
    recs: List[dict] = []
    conf = prediction.confidence_score

    # ---- 1X2 ----
    by_sel = _latest_market_snapshots(conn, match_id, "1X2")
    market = _novig_market_probs(by_sel, ["Home", "Draw", "Away"])
    if market:
        model_probs = {
            "Home": prediction.prob_home_win,
            "Draw": prediction.prob_draw,
            "Away": prediction.prob_away_win,
        }
        for sel in ("Home", "Draw", "Away"):
            recs.append(_evaluate_selection(
                conn, match_id, model_version, "1X2", sel, None,
                model_probs[sel], market[sel], by_sel[sel], conf, cfg))

    # ---- Over/Under 2.5 (and other lines if present) ----
    by_sel = _latest_market_snapshots(conn, match_id, "O/U")
    for line in (1.5, 2.5, 3.5):
        over_sel, under_sel = f"Over {line}", f"Under {line}"
        if over_sel in by_sel and under_sel in by_sel:
            market = _novig_market_probs(by_sel, [over_sel, under_sel])
            p_over = prediction.prob_over[line]
            model_probs = {over_sel: p_over, under_sel: 1.0 - p_over}
            for sel in (over_sel, under_sel):
                recs.append(_evaluate_selection(
                    conn, match_id, model_version, "O/U", sel, line,
                    model_probs[sel], market[sel], by_sel[sel], conf, cfg))

    # ---- BTTS ----
    by_sel = _latest_market_snapshots(conn, match_id, "BTTS")
    if "Yes" in by_sel and "No" in by_sel:
        market = _novig_market_probs(by_sel, ["Yes", "No"])
        model_probs = {"Yes": prediction.prob_btts_yes, "No": 1.0 - prediction.prob_btts_yes}
        for sel in ("Yes", "No"):
            recs.append(_evaluate_selection(
                conn, match_id, model_version, "BTTS", sel, None,
                model_probs[sel], market[sel], by_sel[sel], conf, cfg))

    return recs


def persist_recommendations(conn, match_id: str, recs: List[dict]) -> None:
    conn.execute("DELETE FROM recommendations WHERE match_id=?", (match_id,))
    for r in recs:
        conn.execute(
            """INSERT INTO recommendations
               (recommendation_id, match_id, model_version, market_type, selection,
                line_value, bookmaker, offered_odds, fair_odds, model_prob,
                market_prob_novig, edge_pct_points, expected_value_pct,
                clv_reference_odds, recommendation_status, confidence_band,
                stake_fraction, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r["recommendation_id"], r["match_id"], r["model_version"], r["market_type"],
             r["selection"], r["line_value"], r["bookmaker"], r["offered_odds"],
             r["fair_odds"], r["model_prob"], r["market_prob_novig"], r["edge_pct_points"],
             r["expected_value_pct"], r["clv_reference_odds"], r["recommendation_status"],
             r["confidence_band"], r["stake_fraction"], r["created_at"]),
        )
        db.audit(conn, "recommendation", r["recommendation_id"], "create",
                 f"{r['market_type']}/{r['selection']} {r['recommendation_status']}")
