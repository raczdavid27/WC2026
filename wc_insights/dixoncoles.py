"""Dixon-Coles attack/defense goals model (suggestions #1 and #4).

A bivariate-Poisson-style model where each team has an attack and a defense
rating, plus a global home advantage and a low-score dependence parameter rho:

    log lambda_home = home_adv*(played at home) + att[home] - def[away]
    log lambda_away =                              att[away] - def[home]

Goals are Poisson with those means, with the Dixon-Coles tau correction applied
to the four low-scoring scorelines (0-0, 1-0, 0-1, 1-1) to fix the draw / low-
total probabilities that independent Poisson gets wrong.

Fitting is two-step (the standard practical approach), pure numpy + a hand-rolled
Adam optimizer (no scipy):
    1. Fit att / def / home_adv by weighted Poisson MLE (gradient ascent).
    2. Fit rho by a 1-D search holding the marginals fixed.
Matches are weighted by exponential time decay so recent form dominates.
"""

import json
import math
import os
from datetime import datetime, timezone

import numpy as np

ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_params.json")
MAX_GOALS = 10


# --------------------------------------------------------------------------
# tau correction
# --------------------------------------------------------------------------
def tau(h, a, lam, mu, rho):
    if h == 0 and a == 0:
        return 1.0 - lam * mu * rho
    if h == 0 and a == 1:
        return 1.0 + lam * rho
    if h == 1 and a == 0:
        return 1.0 + mu * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit(df, teams, half_life_days: float = 540.0, iters: int = 600, lr: float = 0.05):
    """Fit the model on a normalized matches DataFrame (see historical.load).

    `teams` is the index universe; matches involving teams outside it are dropped.
    Returns a params dict (also the on-disk artifact schema).
    """
    idx = {t: i for i, t in enumerate(teams)}
    mask = df["home"].isin(idx) & df["away"].isin(idx)
    d = df[mask]
    n = len(teams)

    h = d["home"].map(idx).to_numpy()
    a = d["away"].map(idx).to_numpy()
    hg = d["hg"].to_numpy(dtype=float)
    ag = d["ag"].to_numpy(dtype=float)
    not_neutral = (~d["neutral"].to_numpy()).astype(float)
    w = np.power(0.5, d["age_days"].to_numpy() / half_life_days)  # time decay
    if "importance" in d.columns:  # up-weight e.g. live World Cup results
        w = w * d["importance"].fillna(1.0).to_numpy()

    att = np.zeros(n)
    dfn = np.zeros(n)
    home_adv = 0.25

    # --- Adam over att / def / home_adv (Poisson log-likelihood) ---
    params = [att, dfn]
    m = [np.zeros(n), np.zeros(n), 0.0]
    v = [np.zeros(n), np.zeros(n), 0.0]
    b1, b2, eps = 0.9, 0.999, 1e-8
    ha = home_adv
    for t in range(1, iters + 1):
        lam = np.exp(ha * not_neutral + att[h] - dfn[a])
        mu = np.exp(att[a] - dfn[h])
        # residuals
        rh = w * (hg - lam)   # d/d(log lam)
        ra = w * (ag - mu)    # d/d(log mu)
        g_att = np.zeros(n)
        g_dfn = np.zeros(n)
        np.add.at(g_att, h, rh)          # att[home] in lam
        np.add.at(g_att, a, ra)          # att[away] in mu
        np.add.at(g_dfn, a, -rh)         # -def[away] in lam
        np.add.at(g_dfn, h, -ra)         # -def[home] in mu
        g_ha = np.sum(rh * not_neutral)

        for i, g in enumerate((g_att, g_dfn)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mhat = m[i] / (1 - b1 ** t)
            vhat = v[i] / (1 - b2 ** t)
            params[i] += lr * mhat / (np.sqrt(vhat) + eps)
        m[2] = b1 * m[2] + (1 - b1) * g_ha
        v[2] = b2 * v[2] + (1 - b2) * g_ha * g_ha
        ha += lr * (m[2] / (1 - b1 ** t)) / (math.sqrt(v[2] / (1 - b2 ** t)) + eps)

        att, dfn = params
        att -= att.mean()  # identifiability: mean attack = 0

    # --- fit rho on low-score cells, marginals fixed ---
    lam = np.exp(ha * not_neutral + att[h] - dfn[a])
    mu = np.exp(att[a] - dfn[h])
    rho = _fit_rho(hg, ag, lam, mu, w)

    return {
        "version": "dixon-coles-1.0.0",
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_matches": int(len(d)),
        "half_life_days": half_life_days,
        "home_adv": float(ha),
        "rho": float(rho),
        "attack": {teams[i]: float(att[i]) for i in range(n)},
        "defense": {teams[i]: float(dfn[i]) for i in range(n)},
    }


def _fit_rho(hg, ag, lam, mu, w):
    """1-D search for rho maximizing the weighted low-score log-likelihood."""
    low = (hg <= 1) & (ag <= 1)
    H, A, L, M, W = hg[low].astype(int), ag[low].astype(int), lam[low], mu[low], w[low]

    def nll(rho):
        t = np.ones_like(L)
        t = np.where((H == 0) & (A == 0), 1 - L * M * rho, t)
        t = np.where((H == 0) & (A == 1), 1 + L * rho, t)
        t = np.where((H == 1) & (A == 0), 1 + M * rho, t)
        t = np.where((H == 1) & (A == 1), 1 - rho, t)
        if np.any(t <= 0):
            return -np.inf
        return np.sum(W * np.log(t))

    best, best_ll = 0.0, nll(0.0)
    for rho in np.linspace(-0.2, 0.2, 81):
        ll = nll(rho)
        if ll > best_ll:
            best, best_ll = float(rho), ll
    return best


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def lambdas(params, home, away, neutral):
    """Return (lambda_home, lambda_away) or None if a team is unrated."""
    att, dfn = params["attack"], params["defense"]
    if home not in att or away not in att:
        return None
    ha = 0.0 if neutral else params["home_adv"]
    lam = math.exp(ha + att[home] - dfn[away])
    mu = math.exp(att[away] - dfn[home])
    return max(0.05, lam), max(0.05, mu)


def score_matrix(lam, mu, rho, max_goals=MAX_GOALS):
    ph = np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(max_goals + 1)])
    pa = np.array([math.exp(-mu) * mu ** k / math.factorial(k) for k in range(max_goals + 1)])
    M = np.outer(ph, pa)
    M[0, 0] *= tau(0, 0, lam, mu, rho)
    M[0, 1] *= tau(0, 1, lam, mu, rho)
    M[1, 0] *= tau(1, 0, lam, mu, rho)
    M[1, 1] *= tau(1, 1, lam, mu, rho)
    return M / M.sum()


def markets(lam, mu, rho, over_lines=(1.5, 2.5, 3.5)):
    """Derive 1X2, totals, and BTTS probabilities from the score matrix."""
    M = score_matrix(lam, mu, rho)
    n = M.shape[0]
    p_home = float(np.tril(M, -1).sum())   # home goals > away goals
    p_away = float(np.triu(M, 1).sum())
    p_draw = float(np.trace(M))
    totals = np.add.outer(np.arange(n), np.arange(n))
    over = {ln: float(M[totals > ln].sum()) for ln in over_lines}
    btts = float(M[1:, 1:].sum())
    return {
        "prob_home_win": p_home, "prob_draw": p_draw, "prob_away_win": p_away,
        "prob_over": over, "prob_btts_yes": btts,
    }


# --------------------------------------------------------------------------
# Artifact I/O
# --------------------------------------------------------------------------
def save(params, path=ARTIFACT_PATH):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(params, fh)


def load(path=ARTIFACT_PATH):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
