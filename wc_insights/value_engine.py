"""Value engine — the auditable betting math.

All functions are pure (no I/O) so they can be unit-tested in isolation, per
the spec's requirement to cover no-vig conversion, EV, fair odds, edge, and CLV.

Definitions (see spec "Mathematical definitions"):
    raw implied prob   p = 1 / o
    no-vig prob        p'_i = p_i / sum(p)
    edge               edge = p_model - p_market_novig
    expected value     EV = p_model * o - 1
    fair odds          fair = 1 / p_model
    closing line value CLV = placed_odds / closing_odds - 1   (price-improvement
                       convention; positive means you beat the close)
"""

from typing import Dict, List, Sequence


def implied_prob_raw(decimal_odds: float) -> float:
    """Raw implied probability from decimal odds. Includes bookmaker margin."""
    if decimal_odds is None or decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds!r}")
    return 1.0 / decimal_odds


def overround(decimal_odds: Sequence[float]) -> float:
    """Bookmaker overround (margin) of a market = sum(1/o) - 1."""
    return sum(implied_prob_raw(o) for o in decimal_odds) - 1.0


def novig_probs(decimal_odds: Sequence[float]) -> List[float]:
    """No-vig probabilities for a complete market.

    Normalizes raw implied probabilities so they sum to 1, removing the
    overround. Works for 2-way (O/U, BTTS, DNB) and 3-way (1X2) markets.
    """
    raw = [implied_prob_raw(o) for o in decimal_odds]
    total = sum(raw)
    if total <= 0:
        raise ValueError("sum of implied probabilities must be positive")
    probs = [p / total for p in raw]
    for p in probs:
        if p <= 0.0 or p >= 1.0:
            raise ValueError(f"impossible no-vig probability: {p}")
    return probs


def novig_prob_for(decimal_odds: Sequence[float], index: int) -> float:
    """No-vig probability of a single selection within its market."""
    return novig_probs(decimal_odds)[index]


def edge(model_prob: float, market_prob_novig: float) -> float:
    """Probability edge as a fraction (model minus no-vig market)."""
    return model_prob - market_prob_novig


def edge_pct_points(model_prob: float, market_prob_novig: float) -> float:
    """Edge expressed in percentage points for UI readability."""
    return edge(model_prob, market_prob_novig) * 100.0


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """EV per unit staked = p*o - 1. Returned as a fraction (0.05 == +5%)."""
    return model_prob * decimal_odds - 1.0


def expected_value_pct(model_prob: float, decimal_odds: float) -> float:
    return expected_value(model_prob, decimal_odds) * 100.0


def fair_odds(model_prob: float) -> float:
    """Fair decimal odds = 1 / model probability."""
    if model_prob <= 0.0:
        raise ValueError("model probability must be positive")
    return 1.0 / model_prob


def closing_line_value(placed_odds: float, closing_odds: float) -> float:
    """CLV as a fraction using the price-improvement convention.

    CLV = placed_odds / closing_odds - 1. Positive means the price you took
    was better (higher) than the closing price, i.e. you beat the close.
    This single definition is used consistently across reports and dashboards.
    """
    if placed_odds <= 1.0 or closing_odds <= 1.0:
        raise ValueError("odds must be > 1.0 for CLV")
    return placed_odds / closing_odds - 1.0


def closing_line_value_pct(placed_odds: float, closing_odds: float) -> float:
    return closing_line_value(placed_odds, closing_odds) * 100.0


def kelly_fraction(model_prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly stake (as a fraction of bankroll).

    Full Kelly f* = (b*p - q) / b, where b = o - 1, q = 1 - p. We scale by
    `fraction` (default quarter-Kelly) and floor negatives at 0 — Kelly is
    unstable under model uncertainty, so the product never recommends a
    negative/levered stake.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    f_star = (b * model_prob - q) / b
    return max(0.0, f_star * fraction)


def pnl_units(result_status: str, stake_units: float, decimal_odds: float) -> float:
    """Profit/loss in units for a settled bet at flat-stake `stake_units`."""
    status = (result_status or "").lower()
    if status == "win":
        return stake_units * (decimal_odds - 1.0)
    if status == "loss":
        return -stake_units
    return 0.0  # void / push


def settle_pnl(result_status: str, decimal_odds: float, stake: float = 1.0) -> float:
    return pnl_units(result_status, stake, decimal_odds)


def best_odds(snapshots: List[Dict]) -> Dict:
    """Pick the snapshot offering the best (highest) decimal odds."""
    return max(snapshots, key=lambda s: s["decimal_odds"])
