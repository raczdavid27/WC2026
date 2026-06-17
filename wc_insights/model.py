"""Baseline expected-goals model (Poisson + Elo), pure Python.

This is the transparent, auditable baseline the spec asks for: an Elo-driven
supremacy estimate combined with recent-xG form, turned into per-side expected
goals (lambda_home, lambda_away), then a Poisson score grid that yields every
modeled market (1X2, totals, BTTS). No numpy required.

The baseline is intentionally interpretable so it can serve as the stable
benchmark even if a heavier production model is added later.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

# --- model constants (tunable) ---------------------------------------------
LEAGUE_TEAM_GOALS = 1.35     # avg goals per team in a neutral match
HOME_ADV_ELO = 60.0          # Elo bump applied to host / quasi-home side
SUPREMACY_SCALE = 2.6        # maps Elo win-expectancy delta -> goal supremacy
FORM_WEIGHT = 0.35           # blend weight of recent xG form vs Elo baseline
MAX_GOALS = 10               # truncation of the Poisson score grid


def elo_expectancy(elo_a: float, elo_b: float, home_adv: float = 0.0) -> float:
    """Classic Elo win expectancy for side A (0..1)."""
    return 1.0 / (1.0 + 10 ** (-((elo_a + home_adv) - elo_b) / 400.0))


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


@dataclass
class MatchInputs:
    """Everything the model needs to score one fixture."""
    elo_home: float
    elo_away: float
    home_field: bool = False           # host / quasi-home advantage for home side
    # team identity (enables the Dixon-Coles attack/defense path; None -> Elo)
    home_name: str = None
    away_name: str = None
    neutral: bool = True
    # recent-form xG (per match); None -> fall back to league average
    form_xg_for_home: float = None
    form_xg_against_home: float = None
    form_xg_for_away: float = None
    form_xg_against_away: float = None
    # opponent-adjusted, shrinkage-weighted form adjustments in log-lambda space
    # (suggestion #5); 0.0 = no recent-form signal yet
    form_att_adj_home: float = 0.0
    form_def_adj_home: float = 0.0
    form_att_adj_away: float = 0.0
    form_def_adj_away: float = 0.0
    # tournament context
    rest_days_home: int = None
    rest_days_away: int = None
    travel_km_home: float = None
    travel_km_away: float = None
    must_win_home: bool = False
    must_win_away: bool = False
    # availability / uncertainty -> confidence score
    injury_count_home: int = 0
    injury_count_away: int = 0
    lineup_confirmed: bool = False
    data_freshness_hours: float = 24.0
    missing_data_ratio: float = 0.0


@dataclass
class MatchPrediction:
    lambda_home: float
    lambda_away: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    prob_over: Dict[float, float]      # keyed by line, e.g. 2.5
    prob_btts_yes: float
    confidence_score: float
    feature_contrib: Dict[str, float] = field(default_factory=dict)


def _expected_lambdas(inp: MatchInputs) -> Tuple[float, float, Dict[str, float]]:
    """Derive (lambda_home, lambda_away) and a feature-contribution map."""
    home_adv = HOME_ADV_ELO if inp.home_field else 0.0

    # 1) Elo win expectancy -> goal supremacy.
    we_home = elo_expectancy(inp.elo_home, inp.elo_away, home_adv)
    supremacy = (we_home - 0.5) * 2.0 * SUPREMACY_SCALE  # +ve favours home

    # 2) Total-goals expectation: league base, nudged by combined attacking form.
    base_total = 2.0 * LEAGUE_TEAM_GOALS
    form_total_adj = 0.0
    if None not in (inp.form_xg_for_home, inp.form_xg_for_away):
        avg_att = (inp.form_xg_for_home + inp.form_xg_for_away) / 2.0
        form_total_adj = FORM_WEIGHT * (avg_att - LEAGUE_TEAM_GOALS)
    total = max(1.4, base_total + form_total_adj)

    # 3) Context nudges to supremacy (small, bounded).
    rest_adj = 0.0
    if None not in (inp.rest_days_home, inp.rest_days_away):
        rest_adj = 0.04 * (inp.rest_days_home - inp.rest_days_away)
    travel_adj = 0.0
    if None not in (inp.travel_km_home, inp.travel_km_away):
        travel_adj = -0.00005 * (inp.travel_km_home - inp.travel_km_away)
    motivation_adj = 0.0
    if inp.must_win_home and not inp.must_win_away:
        motivation_adj += 0.12
    if inp.must_win_away and not inp.must_win_home:
        motivation_adj -= 0.12

    supremacy_adj = supremacy + rest_adj + travel_adj + motivation_adj

    lam_home = max(0.15, (total + supremacy_adj) / 2.0)
    lam_away = max(0.15, (total - supremacy_adj) / 2.0)

    # 4) Blend in defensive form: a leaky defence raises opponent lambda.
    if None not in (inp.form_xg_against_away, inp.form_xg_against_home):
        def_home = inp.form_xg_against_home - LEAGUE_TEAM_GOALS
        def_away = inp.form_xg_against_away - LEAGUE_TEAM_GOALS
        lam_home = max(0.15, lam_home + FORM_WEIGHT * def_away * 0.5)
        lam_away = max(0.15, lam_away + FORM_WEIGHT * def_home * 0.5)

    contrib = {
        "elo_supremacy": round(supremacy, 3),
        "form_total_adj": round(form_total_adj, 3),
        "rest_adj": round(rest_adj, 3),
        "travel_adj": round(travel_adj, 3),
        "motivation_adj": round(motivation_adj, 3),
    }
    return lam_home, lam_away, contrib


def _confidence(inp: MatchInputs) -> float:
    """Map availability/freshness signals to a 0..1 reliability score."""
    score = 1.0
    score -= min(0.30, 0.05 * (inp.injury_count_home + inp.injury_count_away))
    if not inp.lineup_confirmed:
        score -= 0.10
    if inp.data_freshness_hours is not None:
        score -= min(0.25, max(0.0, (inp.data_freshness_hours - 6.0) / 48.0))
    score -= min(0.25, inp.missing_data_ratio)
    return max(0.05, min(1.0, score))


# --- Dixon-Coles artifact (lazy-loaded, cached) ----------------------------
_DC_PARAMS = "unloaded"


def _dc_params():
    global _DC_PARAMS
    if _DC_PARAMS == "unloaded":
        from . import dixoncoles
        _DC_PARAMS = dixoncoles.load()
    return _DC_PARAMS


def reload_artifacts():
    """Force re-read of fitted artifacts (call after `fit`)."""
    global _DC_PARAMS
    _DC_PARAMS = "unloaded"


def dc_info() -> dict:
    """Summary of the loaded rating artifact (for API metadata)."""
    p = _dc_params()
    if not p:
        return {}
    return {"ratings_fitted_at": p.get("fitted_at"),
            "wc_matches_in_ratings": p.get("n_wc_matches", 0)}


def _context_form_adjust(inp: MatchInputs):
    """Per-side log-lambda adjustments from context + opponent-adjusted form."""
    context = 0.0
    if None not in (inp.rest_days_home, inp.rest_days_away):
        context += 0.04 * (inp.rest_days_home - inp.rest_days_away)
    if None not in (inp.travel_km_home, inp.travel_km_away):
        context += -0.00005 * (inp.travel_km_home - inp.travel_km_away)
    if inp.must_win_home and not inp.must_win_away:
        context += 0.12
    if inp.must_win_away and not inp.must_win_home:
        context -= 0.12

    # form_att/def adjustments already arrive opponent-adjusted + shrunk (#5)
    adj_home = inp.form_att_adj_home + 0.5 * inp.form_def_adj_away + 0.15 * context
    adj_away = inp.form_att_adj_away + 0.5 * inp.form_def_adj_home - 0.15 * context
    clip = lambda x: max(-0.5, min(0.5, x))
    contrib = {
        "context_logadj": round(0.15 * context, 3),
        "form_att_home": round(inp.form_att_adj_home, 3),
        "form_att_away": round(inp.form_att_adj_away, 3),
    }
    return clip(adj_home), clip(adj_away), contrib


def _markets_from_grid(lam_home, lam_away):
    """Independent-Poisson markets (Elo fallback path)."""
    ph = [poisson_pmf(k, lam_home) for k in range(MAX_GOALS + 1)]
    pa = [poisson_pmf(k, lam_away) for k in range(MAX_GOALS + 1)]
    p_home = p_draw = p_away = p_btts = 0.0
    over_lines = [1.5, 2.5, 3.5]
    p_over = {ln: 0.0 for ln in over_lines}
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = ph[h] * pa[a]
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
            if h >= 1 and a >= 1:
                p_btts += p
            for ln in over_lines:
                if h + a > ln:
                    p_over[ln] += p
    s = p_home + p_draw + p_away
    return {"prob_home_win": p_home / s, "prob_draw": p_draw / s, "prob_away_win": p_away / s,
            "prob_over": p_over, "prob_btts_yes": p_btts}


def predict(inp: MatchInputs) -> MatchPrediction:
    params = _dc_params()
    dc_res = None
    if params and inp.home_name and inp.away_name:
        from . import dixoncoles
        lm = dixoncoles.lambdas(params, inp.home_name, inp.away_name, inp.neutral)
        if lm:
            adj_h, adj_a, contrib = _context_form_adjust(inp)
            lam_home = max(0.1, lm[0] * math.exp(adj_h))
            lam_away = max(0.1, lm[1] * math.exp(adj_a))
            contrib.update({"dc_attack_home": round(params["attack"].get(inp.home_name, 0), 3),
                            "dc_attack_away": round(params["attack"].get(inp.away_name, 0), 3),
                            "model": "dixon-coles"})
            mk = dixoncoles.markets(lam_home, lam_away, params["rho"])
            dc_res = (lam_home, lam_away, mk, contrib)

    if dc_res:
        lam_home, lam_away, mk, contrib = dc_res
    else:
        lam_home, lam_away, contrib = _expected_lambdas(inp)
        contrib["model"] = "elo-poisson"
        mk = _markets_from_grid(lam_home, lam_away)

    return MatchPrediction(
        lambda_home=lam_home,
        lambda_away=lam_away,
        prob_home_win=mk["prob_home_win"],
        prob_draw=mk["prob_draw"],
        prob_away_win=mk["prob_away_win"],
        prob_over=mk["prob_over"],
        prob_btts_yes=mk["prob_btts_yes"],
        confidence_score=_confidence(inp),
        feature_contrib=contrib,
    )


def confidence_band(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"
