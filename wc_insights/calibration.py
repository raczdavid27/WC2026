"""Calibration & performance metrics.

Short tournament samples make raw ROI misleading, so the spec mandates
calibration (Brier, log loss, reliability buckets) and CLV alongside ROI.
The metric functions are pure so they can be tested independently.

This module also holds the *calibration layer* (suggestion #2): isotonic and
Platt recalibrators that map raw model probabilities to calibrated ones, fit
out-of-sample from the backtest, and applied before recommendations.
"""

import json
import math
import os
from typing import Dict, List, Tuple

EPS = 1e-12
PROB_FLOOR = 1e-4  # clamp calibrated probabilities away from 0/1 (keeps 1/p finite)
CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration.json")


def brier_score(pairs: List[Tuple[float, int]]) -> float:
    """Mean squared error of probabilistic predictions. pairs = [(p, outcome)]."""
    if not pairs:
        return 0.0
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def log_loss(pairs: List[Tuple[float, int]]) -> float:
    """Binary log loss; probabilities clamped to avoid infinities."""
    if not pairs:
        return 0.0
    total = 0.0
    for p, o in pairs:
        p = min(1 - EPS, max(EPS, p))
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(pairs)


def reliability_buckets(pairs: List[Tuple[float, int]], width: float = 0.05) -> List[dict]:
    """Group predictions into probability bands and compare predicted vs actual.

    Returns one row per non-empty bucket: predicted mean, observed frequency,
    and count — the data behind a reliability diagram.
    """
    buckets: Dict[int, List[Tuple[float, int]]] = {}
    n = int(round(1.0 / width))
    for p, o in pairs:
        idx = min(n - 1, int(p / width))
        buckets.setdefault(idx, []).append((p, o))
    rows = []
    for idx in sorted(buckets):
        grp = buckets[idx]
        lo, hi = idx * width, (idx + 1) * width
        pred_mean = sum(p for p, _ in grp) / len(grp)
        obs_freq = sum(o for _, o in grp) / len(grp)
        rows.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "lo": lo, "hi": hi,
            "predicted_mean": pred_mean,
            "observed_freq": obs_freq,
            "count": len(grp),
        })
    return rows


def calibration_error(pairs: List[Tuple[float, int]], width: float = 0.05) -> float:
    """Expected Calibration Error: count-weighted |predicted - observed|."""
    rows = reliability_buckets(pairs, width)
    total = sum(r["count"] for r in rows)
    if total == 0:
        return 0.0
    return sum(r["count"] * abs(r["predicted_mean"] - r["observed_freq"]) for r in rows) / total


# --------------------------------------------------------------------------
# Recalibrators (suggestion #2)
# --------------------------------------------------------------------------
def _pava(xs: List[float], ys: List[int]):
    """Pool-Adjacent-Violators: monotone non-decreasing fit of y on sorted x.

    Returns (thresholds, values) defining a step/interp function. Pure Python.
    """
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    sx = [xs[i] for i in order]
    # blocks: [sum_y, weight, value]
    blocks = [[float(ys[i]), 1.0, float(ys[i])] for i in order]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i + 1][2] + 1e-12:
            s = blocks[i][0] + blocks[i + 1][0]
            w = blocks[i][1] + blocks[i + 1][1]
            blocks[i:i + 2] = [[s, w, s / w]]
            if i > 0:
                i -= 1
        else:
            i += 1
    # expand block values back to per-point, aligned to sorted x
    vals, bi, used = [], 0, 0.0
    for b in blocks:
        for _ in range(int(round(b[1]))):
            vals.append(b[2])
    return sx, vals


class IsotonicCalibrator:
    """Monotone, non-parametric recalibration via PAVA with linear interp."""

    def __init__(self, thresholds=None, values=None):
        self.thresholds = thresholds or []
        self.values = values or []

    @classmethod
    def fit(cls, pairs: List[Tuple[float, int]]):
        if len(pairs) < 10:
            return cls([0.0, 1.0], [0.0, 1.0])  # identity-ish
        xs = [p for p, _ in pairs]
        ys = [y for _, y in pairs]
        sx, vals = _pava(xs, ys)
        # collapse duplicate x to last value (monotone)
        tx, tv = [], []
        for x, v in zip(sx, vals):
            if tx and abs(x - tx[-1]) < 1e-9:
                tv[-1] = v
            else:
                tx.append(x)
                tv.append(v)
        return cls(tx, tv)

    def predict(self, p: float) -> float:
        tx, tv = self.thresholds, self.values
        if not tx:
            val = p
        elif p <= tx[0]:
            val = tv[0]
        elif p >= tx[-1]:
            val = tv[-1]
        else:
            # binary search + linear interpolation
            lo, hi = 0, len(tx) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if tx[mid] <= p:
                    lo = mid
                else:
                    hi = mid
            span = tx[hi] - tx[lo]
            val = tv[hi] if span < 1e-12 else tv[lo] + (p - tx[lo]) / span * (tv[hi] - tv[lo])
        # Never emit exactly 0 or 1: isotonic can floor a rare outcome to 0,
        # which would break fair-odds (1/p) and log-loss downstream.
        return min(1.0 - PROB_FLOOR, max(PROB_FLOOR, val))

    def to_dict(self):
        return {"type": "isotonic", "thresholds": self.thresholds, "values": self.values}


class MarketCalibrators:
    """Per-market recalibrators (1X2 / O/U / BTTS) fit from backtest OOS pairs."""

    def __init__(self, cals: Dict[str, IsotonicCalibrator] = None, metrics: dict = None):
        self.cals = cals or {}
        self.metrics = metrics or {}

    @classmethod
    def fit_from_oos(cls, oos: Dict[str, List[Tuple[float, int]]]):
        cals, metrics = {}, {}
        for market, pairs in oos.items():
            cals[market] = IsotonicCalibrator.fit(pairs)
            after = [(cals[market].predict(p), y) for p, y in pairs]
            metrics[market] = {
                "n": len(pairs),
                "log_loss_before": round(log_loss(pairs), 4),
                "log_loss_after": round(log_loss(after), 4),
                "ece_before": round(calibration_error(pairs), 4),
                "ece_after": round(calibration_error(after), 4),
            }
        return cls(cals, metrics)

    def calibrate_markets(self, mk: dict) -> dict:
        """Apply calibration to a model markets dict, renormalizing 1X2."""
        out = dict(mk)
        if "1x2" in self.cals:
            c = self.cals["1x2"]
            h = c.predict(mk["prob_home_win"])
            d = c.predict(mk["prob_draw"])
            a = c.predict(mk["prob_away_win"])
            s = h + d + a or 1.0
            out["prob_home_win"], out["prob_draw"], out["prob_away_win"] = h / s, d / s, a / s
        if "ou25" in self.cals and "prob_over" in mk:
            c = self.cals["ou25"]
            out["prob_over"] = {ln: c.predict(v) for ln, v in mk["prob_over"].items()}
        if "btts" in self.cals:
            out["prob_btts_yes"] = self.cals["btts"].predict(mk["prob_btts_yes"])
        return out

    def save(self, path=CALIBRATION_PATH):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"cals": {k: v.to_dict() for k, v in self.cals.items()},
                       "metrics": self.metrics}, fh)

    @classmethod
    def load(cls, path=CALIBRATION_PATH):
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        cals = {k: IsotonicCalibrator(v["thresholds"], v["values"]) for k, v in d["cals"].items()}
        return cls(cals, d.get("metrics", {}))
