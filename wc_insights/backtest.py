"""Walk-forward backtest (suggestion #3).

Rolling-origin validation: repeatedly fit the Dixon-Coles model on everything
before a cutoff date, predict the next block of matches strictly out-of-sample,
and score the predictions. This (a) proves the model generalizes and that the
Dixon-Coles tau correction helps, and (b) produces the out-of-sample
(probability, outcome) pairs used to fit the calibration layer (#2).

No future data ever informs a prediction — fits use only matches dated before
the fold cutoff.
"""

import numpy as np
import pandas as pd

from . import dixoncoles as dc, historical
from .calibration import brier_score, log_loss


def _outcome(hg, ag):
    return "H" if hg > ag else "A" if ag > hg else "D"


def walk_forward(df, teams, n_folds: int = 8, test_days: int = 120,
                 half_life_days: float = 540.0):
    """Return (report, oos_pairs).

    oos_pairs: {"1x2": [(p,y)...], "ou25": [...], "btts": [...]} pooled across
    folds — the training set for the calibrator.
    """
    dates = pd.to_datetime(df["date"])
    end = dates.max()
    start = end - pd.Timedelta(days=test_days * n_folds)

    oos = {"1x2": [], "ou25": [], "btts": []}
    # competing scorers, all evaluated on the same OOS matches
    ll = {"dixon_coles": [], "indep_poisson": [], "base_rate": []}
    n_eval = 0

    # global base rates (home/draw/away) from the training prefix
    base = df[dates < start]
    br = base.apply(lambda r: _outcome(r["hg"], r["ag"]), axis=1).value_counts(normalize=True)
    base_p = {k: float(br.get(k, 1 / 3)) for k in "HDA"}

    for f in range(n_folds):
        cut = start + pd.Timedelta(days=test_days * f)
        nxt = cut + pd.Timedelta(days=test_days)
        train = df[dates < cut]
        test = df[(dates >= cut) & (dates < nxt)]
        if len(train) < 500 or test.empty:
            continue
        params = dc.fit(train, teams, half_life_days=half_life_days, iters=400)

        for _, r in test.iterrows():
            lm = dc.lambdas(params, r["home"], r["away"], bool(r["neutral"]))
            if not lm:
                continue
            y = _outcome(r["hg"], r["ag"])
            mk = dc.markets(lm[0], lm[1], params["rho"])
            indep = dc.markets(lm[0], lm[1], 0.0)  # same lambdas, no tau
            p = {"H": mk["prob_home_win"], "D": mk["prob_draw"], "A": mk["prob_away_win"]}
            pi = {"H": indep["prob_home_win"], "D": indep["prob_draw"], "A": indep["prob_away_win"]}

            ll["dixon_coles"].append(-np.log(max(1e-9, p[y])))
            ll["indep_poisson"].append(-np.log(max(1e-9, pi[y])))
            ll["base_rate"].append(-np.log(max(1e-9, base_p[y])))
            n_eval += 1

            for sel in "HDA":
                oos["1x2"].append((p[sel], 1 if y == sel else 0))
            over = (r["hg"] + r["ag"]) > 2.5
            oos["ou25"].append((mk["prob_over"][2.5], 1 if over else 0))
            btts = r["hg"] >= 1 and r["ag"] >= 1
            oos["btts"].append((mk["prob_btts_yes"], 1 if btts else 0))

    report = {
        "n_eval_matches": n_eval,
        "folds": n_folds,
        "log_loss": {k: round(float(np.mean(v)), 4) for k, v in ll.items() if v},
        "brier_1x2": round(brier_score(oos["1x2"]), 4),
        "improvement_vs_indep_poisson_pct": None,
        "improvement_vs_base_rate_pct": None,
    }
    if ll["dixon_coles"] and ll["indep_poisson"]:
        dcll = report["log_loss"]["dixon_coles"]
        report["improvement_vs_indep_poisson_pct"] = round(
            100 * (report["log_loss"]["indep_poisson"] - dcll) / report["log_loss"]["indep_poisson"], 2)
        report["improvement_vs_base_rate_pct"] = round(
            100 * (report["log_loss"]["base_rate"] - dcll) / report["log_loss"]["base_rate"], 2)
    return report, oos


def run(since: str = "2014-01-01", n_folds: int = 8) -> tuple:
    df = historical.load(since=since)
    teams = historical.team_universe(df, min_matches=8)
    return walk_forward(df, teams, n_folds=n_folds)
