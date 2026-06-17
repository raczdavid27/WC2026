"""Orchestration: feature generation -> scoring -> recommendations, plus
settlement and performance aggregation.

This is what the admin endpoints (run-predictions, settle) and the scheduled
jobs call into.
"""

from typing import Dict, List, Optional

from . import db, features, model, recommender, value_engine as ve, calibration
from . import MODEL_VERSION

# calibration artifact, lazy-loaded + cached
_CALIBRATOR = "unloaded"


def _calibrator():
    global _CALIBRATOR
    if _CALIBRATOR == "unloaded":
        _CALIBRATOR = calibration.MarketCalibrators.load()
    return _CALIBRATOR


def reload_artifacts():
    """Re-read fitted artifacts (calibration + model). Call after `fit`."""
    global _CALIBRATOR
    _CALIBRATOR = "unloaded"
    model.reload_artifacts()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_match(conn, match: dict, cfg: dict, model_version: str = MODEL_VERSION) -> dict:
    """Run the full per-match pipeline and persist outputs + recommendations.

    Order: features -> raw model -> calibration (#2) -> market-prior blend (#6)
    -> recommendations. Calibration and blending happen here, before the value
    engine sees the probabilities, per the spec's "calibrated before use" rule.
    """
    inp = features.build_inputs(conn, match)
    feats = features.feature_map(conn, match, inp)
    features.persist_features(conn, match["match_id"], model_version, feats)

    pred = model.predict(inp)
    _calibrate(pred)
    _market_blend(conn, match["match_id"], pred, cfg)

    _persist_model_output(conn, match["match_id"], model_version, pred)
    recs = recommender.generate_for_match(conn, match["match_id"], model_version, pred, cfg)
    recommender.persist_recommendations(conn, match["match_id"], recs)
    return {"match_id": match["match_id"], "recommendations": len(recs)}


def _calibrate(pred) -> None:
    """Apply the fitted recalibration layer (#2) in place."""
    cal = _calibrator()
    if not cal:
        return
    mk = cal.calibrate_markets({
        "prob_home_win": pred.prob_home_win, "prob_draw": pred.prob_draw,
        "prob_away_win": pred.prob_away_win, "prob_over": pred.prob_over,
        "prob_btts_yes": pred.prob_btts_yes,
    })
    pred.prob_home_win = mk["prob_home_win"]
    pred.prob_draw = mk["prob_draw"]
    pred.prob_away_win = mk["prob_away_win"]
    pred.prob_over = mk["prob_over"]
    pred.prob_btts_yes = mk["prob_btts_yes"]


def _market_blend(conn, match_id: str, pred, cfg: dict) -> None:
    """Shrink model probabilities toward the no-vig market (#6), in place.

    Weight comes from config (`market_blend_weight`, default 0 = pure model, so
    edges stay genuine). Raising it makes the model a market-anchored prior:
    fewer, more conservative flags. Tune it by watching CLV in the Performance
    Lab — anchor to the market only as much as actually improves CLV.
    """
    w = cfg.get("market_blend_weight", 0.0)
    if w <= 0:
        return

    def novig(market_type, order):
        by_sel = recommender._latest_market_snapshots(conn, match_id, market_type)
        return recommender._novig_market_probs(by_sel, order)

    m = novig("1X2", ["Home", "Draw", "Away"])
    if m:
        h = (1 - w) * pred.prob_home_win + w * m["Home"]
        d = (1 - w) * pred.prob_draw + w * m["Draw"]
        a = (1 - w) * pred.prob_away_win + w * m["Away"]
        s = h + d + a or 1.0
        pred.prob_home_win, pred.prob_draw, pred.prob_away_win = h / s, d / s, a / s
    ou = novig("O/U", ["Over 2.5", "Under 2.5"])
    if ou and 2.5 in pred.prob_over:
        pred.prob_over[2.5] = (1 - w) * pred.prob_over[2.5] + w * ou["Over 2.5"]
    bt = novig("BTTS", ["Yes", "No"])
    if bt:
        pred.prob_btts_yes = (1 - w) * pred.prob_btts_yes + w * bt["Yes"]


def _persist_model_output(conn, match_id: str, model_version: str, pred) -> None:
    conn.execute(
        "DELETE FROM model_outputs WHERE match_id=? AND model_version=?",
        (match_id, model_version),
    )
    conn.execute(
        """INSERT INTO model_outputs
           (output_id, match_id, model_version, lambda_home, lambda_away,
            prob_home_win, prob_draw, prob_away_win, prob_over_15, prob_over_25,
            prob_over_35, prob_btts_yes, fair_odds_home, fair_odds_draw,
            fair_odds_away, confidence_score, prediction_generated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (db.new_id(), match_id, model_version, pred.lambda_home, pred.lambda_away,
         pred.prob_home_win, pred.prob_draw, pred.prob_away_win,
         pred.prob_over[1.5], pred.prob_over[2.5], pred.prob_over[3.5],
         pred.prob_btts_yes, ve.fair_odds(pred.prob_home_win),
         ve.fair_odds(pred.prob_draw), ve.fair_odds(pred.prob_away_win),
         pred.confidence_score, db.utc_now()),
    )


def run_predictions(conn, cfg: dict, only_scheduled: bool = True) -> dict:
    """Score every (optionally only scheduled) match."""
    q = "SELECT * FROM matches"
    if only_scheduled:
        q += " WHERE status != 'Final'"
    matches = [dict(r) for r in conn.execute(q).fetchall()]
    out = [score_match(conn, m, cfg) for m in matches]
    conn.commit()
    return {"scored": len(out), "matches": out}


# --------------------------------------------------------------------------
# Settlement
# --------------------------------------------------------------------------
def _selection_won(market_type: str, selection: str, line_value, hg: int, ag: int) -> Optional[str]:
    """Return Win/Loss/Void for a settled selection given the final score."""
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


def _closing_odds(conn, match_id: str, market_type: str, selection: str) -> Optional[float]:
    row = conn.execute(
        """SELECT decimal_odds FROM odds_snapshots
           WHERE match_id=? AND market_type=? AND selection=? AND is_closing=1
           ORDER BY captured_at DESC LIMIT 1""",
        (match_id, market_type, selection),
    ).fetchone()
    if row:
        return row["decimal_odds"]
    # fall back to most recent snapshot
    row = conn.execute(
        """SELECT decimal_odds FROM odds_snapshots
           WHERE match_id=? AND market_type=? AND selection=?
           ORDER BY captured_at DESC LIMIT 1""",
        (match_id, market_type, selection),
    ).fetchone()
    return row["decimal_odds"] if row else None


def settle_match(conn, match: dict, stake_units: float = 1.0) -> int:
    """Settle all actionable recommendations for a finished match."""
    hg, ag = match.get("home_goals"), match.get("away_goals")
    if hg is None or ag is None:
        return 0
    recs = conn.execute(
        """SELECT * FROM recommendations
           WHERE match_id=? AND recommendation_status IN ('Bet','Lean')""",
        (match["match_id"],),
    ).fetchall()
    settled = 0
    for r in recs:
        r = dict(r)
        result = _selection_won(r["market_type"], r["selection"], r["line_value"], hg, ag)
        if result is None:
            continue
        closing = _closing_odds(conn, match["match_id"], r["market_type"], r["selection"])
        clv = ve.closing_line_value_pct(r["offered_odds"], closing) if closing else None
        pnl = ve.settle_pnl(result, r["offered_odds"], stake_units)

        conn.execute("DELETE FROM settled_bets WHERE recommendation_id=?", (r["recommendation_id"],))
        conn.execute(
            """INSERT INTO settled_bets
               (settled_bet_id, recommendation_id, match_id, result_status,
                closing_odds, clv_pct, pnl_units, settled_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (db.new_id(), r["recommendation_id"], match["match_id"], result,
             closing, clv, pnl, db.utc_now()),
        )
        if closing is not None:
            conn.execute("UPDATE recommendations SET clv_reference_odds=? WHERE recommendation_id=?",
                         (closing, r["recommendation_id"]))
        settled += 1
    db.audit(conn, "match", match["match_id"], "settle", f"{settled} bets")
    return settled


def run_settlement(conn, stake_units: float = 1.0) -> dict:
    matches = [dict(r) for r in conn.execute(
        "SELECT * FROM matches WHERE status='Final'").fetchall()]
    total = sum(settle_match(conn, m, stake_units) for m in matches)
    conn.commit()
    from . import betlog  # log new pre-kickoff picks + grade finished ones
    log = betlog.sync(conn)
    return {"settled_bets": total, "final_matches": len(matches), "bet_log": log}


# --------------------------------------------------------------------------
# Full refresh orchestration (CLI `refresh` + UI button)
# --------------------------------------------------------------------------
def full_refresh(cfg: dict, db_path: str = None) -> dict:
    """Fit (if needed) -> ingest -> update ratings -> predict -> settle.

    Each step manages its own DB connection so the ingest reset never collides
    with an open handle. Returns a summary for the caller/UI.
    """
    import os
    from . import db as _db, dixoncoles, fitting, live_ingest
    db_path = db_path or _db.DEFAULT_DB_PATH
    summary = {}
    if not os.path.exists(dixoncoles.ARTIFACT_PATH):
        summary["fit"] = fitting.fit_all(db_path=db_path)
    summary["ingest"] = live_ingest.ingest(db_path, reset=True)
    summary["ratings"] = fitting.update_ratings(db_path)
    reload_artifacts()
    conn = _db.connect(db_path)
    try:
        summary["predict"] = run_predictions(conn, cfg, only_scheduled=False)
        summary["settle"] = run_settlement(conn)
    finally:
        conn.close()
    return summary


# --------------------------------------------------------------------------
# Performance aggregation
# --------------------------------------------------------------------------
def _model_prob_outcome_pairs(conn) -> List[tuple]:
    """Build (model_prob, outcome) pairs for calibration over settled bets."""
    rows = conn.execute(
        """SELECT r.model_prob AS p, s.result_status AS res
           FROM settled_bets s JOIN recommendations r
             ON s.recommendation_id = r.recommendation_id
           WHERE s.result_status IN ('Win','Loss')""").fetchall()
    return [(r["p"], 1 if r["res"] == "Win" else 0) for r in rows]


def performance_summary(conn) -> dict:
    rows = [dict(r) for r in conn.execute(
        """SELECT s.*, r.market_type, r.confidence_band, r.edge_pct_points,
                  r.stake_fraction, m.stage
           FROM settled_bets s
           JOIN recommendations r ON s.recommendation_id = r.recommendation_id
           JOIN matches m ON s.match_id = m.match_id""").fetchall()]

    def agg(subset: List[dict]) -> dict:
        decided = [x for x in subset if x["result_status"] in ("Win", "Loss")]
        n = len(decided)
        wins = sum(1 for x in decided if x["result_status"] == "Win")
        staked = sum((x["stake_fraction"] or 1.0) for x in decided)
        pnl = sum(x["pnl_units"] or 0.0 for x in decided)
        clvs = [x["clv_pct"] for x in subset if x["clv_pct"] is not None]
        edges = [x["edge_pct_points"] for x in subset if x["edge_pct_points"] is not None]
        return {
            "total_bets": n,
            "win_rate": (wins / n) if n else 0.0,
            "roi_pct": (pnl / staked * 100.0) if staked else 0.0,
            "pnl_units": round(pnl, 3),
            "avg_edge_pct": (sum(edges) / len(edges)) if edges else 0.0,
            "avg_clv_pct": (sum(clvs) / len(clvs)) if clvs else 0.0,
        }

    pairs = _model_prob_outcome_pairs(conn)
    by_market: Dict[str, dict] = {}
    by_stage: Dict[str, dict] = {}
    by_conf: Dict[str, dict] = {}
    for key, bucket in (("market_type", by_market), ("stage", by_stage), ("confidence_band", by_conf)):
        seen = sorted({x[key] for x in rows})
        for v in seen:
            bucket[v] = agg([x for x in rows if x[key] == v])

    overall = agg(rows)
    overall.update({
        "log_loss": round(calibration.log_loss(pairs), 4),
        "brier_score": round(calibration.brier_score(pairs), 4),
        "calibration_error": round(calibration.calibration_error(pairs), 4),
        "by_market": by_market,
        "by_stage": by_stage,
        "by_confidence": by_conf,
        "reliability_buckets": calibration.reliability_buckets(pairs),
        "clv_values": [x["clv_pct"] for x in rows if x["clv_pct"] is not None],
        "roi_curve": _roi_curve(rows),
    })
    return overall


def _roi_curve(rows: List[dict]) -> List[dict]:
    """Cumulative P/L ordered by settlement time — feeds the ROI-over-time chart."""
    decided = sorted([x for x in rows if x["result_status"] in ("Win", "Loss")],
                     key=lambda x: x["settled_at"] or "")
    cum = 0.0
    curve = []
    for i, x in enumerate(decided, 1):
        cum += x["pnl_units"] or 0.0
        curve.append({"n": i, "cum_pnl": round(cum, 3), "settled_at": x["settled_at"]})
    return curve
