# FIFA World Cup Betting Insights

A decision-support web app that estimates fair match probabilities, compares
them against bookmaker odds, flags value bets, and tracks whether the model is
beating the market via **calibration** and **closing line value (CLV)** — not
just ROI. Pre-match only (v1), as specified.

> **Not a betting bot.** Probabilities are model estimates, not guarantees.
> Positive expected value does not ensure short-term profit. Use bankroll
> limits and do not chase losses.

The app is **Python + SQLite + a single HTML/JS frontend** — no Node, no database
server. It needs `requests`, plus `numpy` + `pandas` for the rating model. The
schema, REST contract, math, and screens follow the technical spec; the
recommended FastAPI/Postgres/Next.js stack was swapped for stdlib equivalents
(allowed by the spec) so it runs out-of-the-box on this machine.

**It uses the real 2026 FIFA World Cup** — actual schedule, groups, teams,
venues, results, and odds — pulled live from free sources (no API key needed):

| Data | Source |
|---|---|
| Schedule, teams, venues, stages, live scores, results, match stats | **ESPN** hidden API (`site.api.espn.com/.../fifa.world`) |
| Team strength (Elo, prior) | **eloratings.net** (`World.tsv` + `en.teams.tsv`) |
| Historical results (model fitting) | **martj42/international_results** (every men's international since 1872) |
| Odds — 1X2 moneyline + Over/Under 2.5 | **DraftKings**, embedded in the same ESPN feed |

> Knockout fixtures appear automatically as opponents are decided. BTTS and extra
> O/U lines aren't in the free ESPN feed; drop an Odds API key in `config.json`
> (`odds_api_key`) to add multi-book prices + BTTS — the ingestion is wired for it.

## The model

A **Dixon-Coles attack/defense model**, fit by maximum likelihood on ~12k
historical internationals (time-decay weighted), not hand-tuned constants:

- Each team has fitted **attack** and **defense** ratings + a global **home
  advantage**; goals are Poisson with a **Dixon-Coles low-score (tau)
  correction** so draws and low totals are priced correctly.
- **Recent form** nudges the ratings, opponent-adjusted and shrunk toward the
  baseline by sample size (so 1-2 noisy matches don't swing a prediction). xG
  uses a shot-based estimate (`0.09·shots + 0.15·shots_on_target`), since the
  free data feeds don't expose official xG for the 2026 World Cup.
- A **calibration layer** (isotonic regression, fit out-of-sample from the
  backtest) recalibrates probabilities before the value engine sees them — the
  spec's "calibrated before use" requirement.
- An optional **market-prior blend** (`market_blend_weight`) can shrink toward
  the no-vig market; default 0 keeps edges genuine. Tune it by CLV.
- **In-tournament rating updates**: completed World Cup results are folded back
  into the ratings (up-weighted as recent, high-importance matches) on every
  `refresh`, so predictions for upcoming fixtures reflect tournament form. The
  header shows how many WC results are baked into the current ratings.
- A **walk-forward backtest** validates it: out-of-sample it beats the naive
  base rate by ~20% in log-loss. The Elo/Poisson model remains as an automatic
  fallback for any fixture without fitted ratings.

Run `python app.py fit` to (re)fit; artifacts land in `model_params.json`,
`calibration.json`, and `backtest_report.json`.

## Quick start

```bash
python app.py refresh     # ingest REAL WC data -> predict -> settle
python app.py serve       # http://127.0.0.1:5050
```

Then open <http://127.0.0.1:5050>. (`serve` auto-ingests on first run.)
Re-run `python app.py refresh` any time to pull the latest scores/odds.

For an **offline** run with no network, use synthetic data instead:
`python app.py demo` (then `serve`).

Run the tests:

```bash
python app.py test        # 32 unit tests for the core math + model
```

## CLI

| Command | What it does |
|---|---|
| `python app.py fit` | Fit the Dixon-Coles ratings + calibration on historical results; writes model artifacts + a backtest report. |
| `python app.py ingest` | Pull the **real** WC schedule, teams, Elo, results, and odds from ESPN + eloratings into the SQLite DB. |
| `python app.py predict` | Features → calibrated Dixon-Coles scoring → recommendations for every match. |
| `python app.py settle` | Settle finished matches, compute CLV/PnL, refresh performance. |
| `python app.py update` | Refit the ratings folding in completed WC results (in-tournament form update). |
| `python app.py refresh` | `fit` (if needed) → `ingest` → `update` ratings → `predict` → `settle` (the normal update). |
| `python app.py capture` | Append a fresh odds snapshot + re-score (accumulates line history for CLV). |
| `python app.py watch [--interval N]` | Scheduler loop: `capture` every N minutes (default 15) for live line movement / CLV. |
| `python app.py seed` / `demo` | Build **synthetic** offline data instead of live data (no network). |
| `python app.py serve [--port N]` | Start the web app (auto-fits + ingests on first run). |
| `python app.py test` | Run the unit-test suite (53 tests). |

> **Before the tournament kicks off**, the Performance Lab is empty (no settled
> results yet) — it populates as matches finish and you re-run `refresh`.

## Screens

- **Dashboard** — upcoming/recent matches, top value cards, biggest line moves,
  market-confidence distribution, filters by stage/market/status.
- **Recommendations** — every flagged selection with edge, EV, fair vs offered
  odds, status, confidence, and stake; filterable.
- **Match detail** — 1X2 probability bar, expected goals, fair-vs-market table,
  recommendation panel, risk flags, line-movement chart, and feature
  contributions (explainability). For **finished matches**: a full-time panel
  with the score, **goal scorers** (minute + penalty/own-goal markers), **cards**,
  and a per-team **statistics** comparison (shots, on-target, possession,
  corners, fouls, offsides, xG estimate) — all from the ESPN match feed.
- **Performance Lab** — **recommendation results** from the persistent bet log:
  logged / settled / pending counts, wins-losses, win rate, ROI, profit,
  breakdowns by type (Bet/Lean) and market, model reliability (Brier / log-loss),
  cumulative P/L, recently-graded picks, and open picks. CSV/JSON export.

A **↻ Refresh data** button in the header fetches the latest schedule, odds and
results and re-scores everything (same as `python app.py refresh`); all screens,
including the recommendation results, update when it finishes.

### How recommendation results accumulate

Each actionable pick (Bet/Lean) is recorded in a **separate bet log** (`bet_log.db`)
the first time it's flagged — while the match is still upcoming — and graded once
the match finishes. The bet log survives the ingest rebuild, so results build up
over the tournament. Matches that were already finished the first time you ran the
app were never logged (there was no pre-kickoff pick), so results start from your
next refreshes forward.

## Architecture

Modular layout mirroring the spec's ingestion → modeling → API → frontend split:

```
app.py                  CLI entry point
config.json             thresholds + admin key + market blend weight
wc_insights/
  db.py                 SQLite schema + helpers + audit log
  names.py              shared team-name normalization (feed joins)            (unit-tested)
  value_engine.py       no-vig, implied prob, edge, EV, fair odds, CLV, Kelly  (unit-tested)
  dixoncoles.py         attack/defense + tau goals model (MLE, numpy)          (unit-tested)
  model.py              predict(): Dixon-Coles path + Elo/Poisson fallback     (unit-tested)
  historical.py         martj42 results loader (leakage-safe, cached)
  backtest.py           walk-forward validation + out-of-sample pairs
  calibration.py        isotonic recalibrators + Brier/log-loss/reliability    (unit-tested)
  fitting.py            fit orchestration -> model/calibration/report artifacts
  features.py           feature engineering: form shrinkage, opponent-adjust, freshness
  recommender.py        recommendation rules engine + staking
  pipeline.py           orchestration: calibrate + market-blend + settle + performance
  live_ingest.py        ESPN/eloratings ingest, odds capture, opt-in Odds API
  seed.py               synthetic demo data generator
  api.py                JSON response builders
  server.py             stdlib http.server: REST API + static frontend
static/                 index.html, app.js, styles.css (SPA, inline-SVG charts)
tests/                  unit tests (53)
model_params.json       fitted Dixon-Coles ratings (artifact, regenerated by `fit`)
calibration.json        fitted recalibrators (artifact)
backtest_report.json    validation metrics (artifact)
```

## REST API (`/api/v1`)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/matches` | Match cards. Filters: `stage`, `date`, `recommendation_status`, `market_type`. |
| GET | `/matches/{id}` | Full match detail (model, fair vs market, recs, features, line chart). |
| GET | `/recommendations` | Active recommendations. Filters: `status`, `min_edge`, `min_ev`, `bookmaker`, `market_type`. |
| GET | `/performance` | Aggregate evaluation metrics (main settled bets). |
| GET | `/recommendation-stats` | Bet-log recommendation results (logged/settled/win rate/ROI/breakdowns). |
| POST | `/refresh` | Run a full refresh (ingest + re-score). Local UI convenience; single-flight. |
| POST | `/admin/ingest/odds` | Ingest odds snapshots (validated). |
| POST | `/admin/run-predictions` | Trigger scoring + recommendation refresh. |
| POST | `/admin/settle` | Trigger settlement + performance update. |
| POST | `/admin/lineup` | Set injuries / confirmed-XI for a match and re-score it (lineup-sensitive). |

Admin (POST) routes require the `X-Admin-Key` header (default `demo-admin-key`,
set in `config.json`). Example:

```bash
curl -X POST http://127.0.0.1:5050/api/v1/admin/run-predictions -H "X-Admin-Key: demo-admin-key"
```

## The math (auditable, in `value_engine.py`)

- Raw implied probability: `p = 1/o`
- No-vig probability: `p'_i = p_i / Σ p` (1X2, O/U, BTTS)
- Edge: `p_model − p_market_novig` (stored in percentage points)
- Expected value: `EV = p·o − 1`
- Fair odds: `1 / p_model`
- CLV (price-improvement convention): `placed_odds / closing_odds − 1` — positive
  means you beat the close. One definition, used everywhere.
- Staking: flat 1-unit or quarter-Kelly (floored at 0, capped at a bankroll %).

### Model

A transparent baseline: Elo win-expectancy → goal supremacy, blended with recent
xG form and small context nudges (rest, travel, must-win) → per-side expected
goals `λ`. A truncated Poisson score grid then yields every market (1X2, totals,
BTTS). Calibration metrics (Brier, log loss, reliability buckets, ECC) are
computed over settled bets in the Performance Lab.

### Recommendation policy (defaults in `config.json`)

A selection is **Bet** only if edge ≥ 4.0pp **and** EV ≥ 2.0% **and** model
confidence ≥ 0.55. **Lean** if EV is positive but a filter is weak. **Pass**
otherwise. Conservative by design — tournament samples are small.

## Data ingestion (`live_ingest.py`)

`live_ingest.py` pulls the real tournament into the schema; `seed.py` is a
synthetic fallback for offline use. Both write the same tables
(`teams`, `matches`, `team_match_stats`, `odds_snapshots`), so everything
downstream is identical. Team names are normalized across ESPN / eloratings via
`norm()` + alias table so the feeds join cleanly.

## Notes & limitations

- **xG isn't in the free ESPN feed**, so completed-match `xg_for/against` uses a
  transparent shot-based proxy (`0.09·shots + 0.15·shots_on_target`); swap in a
  real xG feed if you have one. Pre-tournament there are no played matches, so
  the model runs on Elo + context until results arrive.
- **Odds are a single book (DraftKings)** with 1X2 + O/U 2.5. BTTS / extra lines
  need an Odds API key (`odds_api_key` in `config.json`).
- The model gives heavy favorites large supremacy (e.g. host Mexico vs a weak
  side), which can push the underdog's expected goals quite low — it's a
  baseline, calibrated against results over time in the Performance Lab.
- Production hardening (real auth, model-version artifacts, scheduled jobs) is
  scaffolded (audit log, admin key, `model_version` columns) but not fully wired
  — see the spec's Phase 3 items.
```
