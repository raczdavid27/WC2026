"""Synthetic World Cup seed data (demo mode).

Generates a self-contained, deterministic dataset that exercises every part of
the app: teams with Elo + recent xG form, group and knockout fixtures (both
already-played and upcoming), multi-bookmaker odds snapshots with opening /
periodic / closing captures and realistic line movement, and standings context.

The same schema is used for real feeds later — only this generator is replaced.
Bookmaker odds are produced from an Elo-only "market" view plus margin and
noise, while the app's model additionally uses recent form and context, so
genuine (but modest) edges arise naturally rather than being hand-planted.
"""

import random
from datetime import datetime, timedelta, timezone

from . import db
from .model import MatchInputs, predict

SEED = 2026
BOOKMAKERS = ["Pinnacle", "Bet365", "Marathon"]
MARGINS = {"Pinnacle": 0.025, "Bet365": 0.06, "Marathon": 0.045}

# "now" for the demo aligns with the tournament window (2026-06-11).
NOW = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)

TEAMS = [
    # name, confederation, fifa_rank, elo, squad_value(€m), coach, host
    ("Argentina",    "CONMEBOL", 1,  2105, 1180, "L. Scaloni",      0),
    ("France",       "UEFA",     2,  2085, 1320, "D. Deschamps",    0),
    ("Spain",        "UEFA",     3,  2060, 1240, "L. de la Fuente", 0),
    ("England",      "UEFA",     4,  2035, 1380, "T. Tuchel",       0),
    ("Brazil",       "CONMEBOL", 5,  2030, 1100, "D. Ancelotti",    0),
    ("Portugal",     "UEFA",     6,  2015, 1090, "R. Martínez",     0),
    ("Netherlands",  "UEFA",     7,  1990, 1010, "R. Koeman",       0),
    ("USA",          "CONCACAF", 16, 1810,  480, "M. Pochettino",   1),
    ("Mexico",       "CONCACAF", 14, 1800,  330, "J. Aguirre",      1),
    ("Canada",       "CONCACAF", 30, 1750,  290, "J. Marsch",       1),
    ("Croatia",      "UEFA",     10, 1955,  420, "Z. Dalić",        0),
    ("Morocco",      "CAF",      12, 1900,  360, "W. Regragui",     0),
    ("Japan",        "AFC",      18, 1840,  300, "H. Moriyasu",     0),
    ("Senegal",      "CAF",      17, 1855,  340, "P. Cissé",        0),
    ("Uruguay",      "CONMEBOL", 11, 1945,  460, "M. Bielsa",       0),
    ("Germany",      "UEFA",     9,  1975,  980, "J. Nagelsmann",   0),
]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _market_probs(rng, elo_h, elo_w, host):
    """Elo-only market view (1X2, over-line, BTTS) + small noise."""
    mi = MatchInputs(elo_home=elo_h, elo_away=elo_w, home_field=bool(host))
    pred = predict(mi)
    # Modest noise so edges land in a realistic single-digit range rather than
    # producing implausible double-digit EVs (esp. on long-priced tail lines).
    jitter = lambda x, s=0.015: max(0.02, min(0.96, x + rng.gauss(0, s)))
    p_home = jitter(pred.prob_home_win)
    p_draw = jitter(pred.prob_draw, 0.010)
    p_away = jitter(pred.prob_away_win)
    s = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
    over = {ln: jitter(pred.prob_over[ln], 0.012) for ln in (1.5, 2.5, 3.5)}
    btts = jitter(pred.prob_btts_yes, 0.012)
    return p_home, p_draw, p_away, over, btts


def _odds_from_probs(probs, margin):
    """Add bookmaker margin to a complete market and invert to decimal odds."""
    return [round(1.0 / (p * (1.0 + margin)), 2) for p in probs]


def _insert_snapshot(conn, match_id, bookmaker, market, selection, line, odds,
                     captured_at, is_open=0, is_close=0):
    raw = 1.0 / odds
    conn.execute(
        """INSERT INTO odds_snapshots
           (odds_snapshot_id, match_id, bookmaker, market_type, selection,
            line_value, decimal_odds, implied_prob_raw, implied_prob_novig,
            captured_at, is_opening, is_closing, source, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (db.new_id(), match_id, bookmaker, market, selection, line, odds, raw, None,
         _iso(captured_at), is_open, is_close, "synthetic", db.utc_now()),
    )


def _emit_odds_for_match(conn, rng, match_id, elo_h, elo_w, host, kickoff, is_final):
    """Create opening/periodic/closing snapshots across bookmakers with drift."""
    # capture schedule relative to kickoff
    offsets = [("open", timedelta(days=-7)), ("p1", timedelta(days=-3)),
               ("p2", timedelta(days=-1))]
    if is_final:
        offsets.append(("close", timedelta(minutes=-10)))
    else:
        offsets.append(("cur", timedelta(hours=-2) if kickoff - NOW < timedelta(hours=6)
                        else NOW - kickoff))  # "current" snapshot at/just before now

    base = _market_probs(rng, elo_h, elo_w, host)
    p_home, p_draw, p_away, over, btts = base
    drift = rng.gauss(0, 0.02)  # net move of home prob open->close

    for i, (tag, off) in enumerate(offsets):
        captured = kickoff + off
        if not is_final and captured > NOW:
            captured = NOW
        frac = i / max(1, len(offsets) - 1)
        dh = drift * frac
        ph = max(0.02, min(0.96, p_home + dh))
        pa = max(0.02, min(0.96, p_away - dh * 0.6))
        pd = max(0.02, 1 - ph - pa)
        s = ph + pd + pa
        oneX2 = [ph / s, pd / s, pa / s]

        is_open = 1 if tag == "open" else 0
        is_close = 1 if tag == "close" else 0
        for bk in BOOKMAKERS:
            m = MARGINS[bk]
            # small per-book price differences
            bk_probs = [max(0.02, x + rng.gauss(0, 0.008)) for x in oneX2]
            ss = sum(bk_probs)
            bk_probs = [x / ss for x in bk_probs]
            o_home, o_draw, o_away = _odds_from_probs(bk_probs, m)
            _insert_snapshot(conn, match_id, bk, "1X2", "Home", None, o_home, captured, is_open, is_close)
            _insert_snapshot(conn, match_id, bk, "1X2", "Draw", None, o_draw, captured, is_open, is_close)
            _insert_snapshot(conn, match_id, bk, "1X2", "Away", None, o_away, captured, is_open, is_close)

            for ln in (1.5, 2.5, 3.5):
                po = max(0.05, min(0.95, over[ln] + rng.gauss(0, 0.01)))
                o_over, o_under = _odds_from_probs([po, 1 - po], m)
                _insert_snapshot(conn, match_id, bk, "O/U", f"Over {ln}", ln, o_over, captured, is_open, is_close)
                _insert_snapshot(conn, match_id, bk, "O/U", f"Under {ln}", ln, o_under, captured, is_open, is_close)

            pb = max(0.05, min(0.95, btts + rng.gauss(0, 0.01)))
            o_yes, o_no = _odds_from_probs([pb, 1 - pb], m)
            _insert_snapshot(conn, match_id, bk, "BTTS", "Yes", None, o_yes, captured, is_open, is_close)
            _insert_snapshot(conn, match_id, bk, "BTTS", "No", None, o_no, captured, is_open, is_close)


def _emit_history(conn, rng, team_id, elo, kickoff_before, n=6):
    """Synthesize recent-form match stats so the model has xG form to use."""
    strength = (elo - 1850) / 100.0  # ~ -1..+2.5
    for i in range(n):
        date = kickoff_before - timedelta(days=7 * (i + 1))
        xg_for = max(0.2, rng.gauss(1.35 + 0.35 * strength, 0.4))
        xg_against = max(0.2, rng.gauss(1.35 - 0.30 * strength, 0.4))
        gf = max(0, round(rng.gauss(xg_for, 0.6)))
        ga = max(0, round(rng.gauss(xg_against, 0.6)))
        result = "W" if gf > ga else "D" if gf == ga else "L"
        conn.execute(
            """INSERT INTO team_match_stats
               (stat_id, match_id, team_id, opponent_team_id, venue_type, result,
                goals_for, goals_against, xg_for, xg_against, shots, shots_on_target,
                big_chances_for, big_chances_against, possession_pct, cards_yellow,
                cards_red, source, match_date, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (db.new_id(), None, team_id, None, "Neutral", result, gf, ga,
             round(xg_for, 2), round(xg_against, 2), round(8 + 4 * strength),
             round(3 + 1.5 * strength), round(1 + strength), max(0, round(2 - strength)),
             round(50 + 6 * strength, 1), rng.randint(0, 3), 0,
             "synthetic", _iso(date), db.utc_now()),
        )


def _final_score(rng, elo_h, elo_w, host):
    mi = MatchInputs(elo_home=elo_h, elo_away=elo_w, home_field=bool(host))
    pred = predict(mi)
    hg = max(0, round(rng.gauss(pred.lambda_home, 1.0)))
    ag = max(0, round(rng.gauss(pred.lambda_away, 1.0)))
    return hg, ag


_SURNAMES = ["Silva", "Müller", "García", "Rossi", "Tanaka", "Mensah", "Novak",
             "Haaland", "Mbappé", "Kane", "Lewa", "Vinícius", "Pedri", "Modrić",
             "Osimhen", "Son", "Álvarez", "Foden", "Wirtz", "Bellingham"]


def _emit_final_details(conn, rng, mid, home_id, away_id, home_name, away_name, hg, ag, kickoff):
    """Synthetic goal scorers + full match stats for a finished demo match."""
    now = db.utc_now()
    order = 0
    used_minutes = set()
    for team_id, goals in ((home_id, hg), (away_id, ag)):
        for _ in range(goals):
            m = rng.randint(2, 92)
            while m in used_minutes:
                m = rng.randint(2, 92)
            used_minutes.add(m)
            etype = "penalty" if rng.random() < 0.12 else "goal"
            conn.execute(
                """INSERT INTO match_events (event_id, match_id, team_id, player_name,
                   minute, event_type, sort_order, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                (db.new_id(), mid, team_id, rng.choice(_SURNAMES), f"{m}'", etype, m, now),
            )
            order += 1
    # per-team stats
    for team_id, gf, ga in ((home_id, hg, ag), (away_id, ag, hg)):
        shots = rng.randint(6, 20)
        sot = rng.randint(max(gf, 2), max(gf + 2, shots // 2 + 1))
        conn.execute(
            """INSERT INTO team_match_stats (stat_id, match_id, team_id, opponent_team_id,
               venue_type, result, goals_for, goals_against, xg_for, xg_against, shots,
               shots_on_target, big_chances_for, big_chances_against, possession_pct,
               cards_yellow, cards_red, corners, fouls, offsides, saves, source,
               match_date, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (db.new_id(), mid, team_id, away_id if team_id == home_id else home_id, "Neutral",
             "W" if gf > ga else "D" if gf == ga else "L", gf, ga,
             round(0.09 * shots + 0.15 * sot, 2), None, shots, sot, None, None,
             round(rng.uniform(40, 60), 1), rng.randint(0, 4), 0 if rng.random() < 0.9 else 1,
             rng.randint(2, 9), rng.randint(6, 18), rng.randint(0, 5), rng.randint(1, 6),
             "synthetic", _iso(kickoff), now),
        )
    # fill xg_against from the opponent's xg_for
    rows = [dict(r) for r in conn.execute(
        "SELECT stat_id, team_id, opponent_team_id, xg_for FROM team_match_stats WHERE match_id=?",
        (mid,)).fetchall()]
    by_team = {r["team_id"]: r["xg_for"] for r in rows}
    for r in rows:
        conn.execute("UPDATE team_match_stats SET xg_against=? WHERE stat_id=?",
                     (by_team.get(r["opponent_team_id"]), r["stat_id"]))


def seed(db_path: str = db.DEFAULT_DB_PATH) -> dict:
    rng = random.Random(SEED)
    conn = db.reset_db(db_path)

    # --- teams ---
    team_ids = {}
    for name, conf, rank, elo, sv, coach, host in TEAMS:
        tid = db.new_id()
        team_ids[name] = (tid, elo, host)
        now = db.utc_now()
        conn.execute(
            """INSERT INTO teams (team_id, team_name, confederation, fifa_rank,
               elo_rating, squad_value, coach_name, host_flag, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (tid, name, conf, rank, elo, sv, coach, host, now, now),
        )

    # --- recent form history for every team ---
    for name, (tid, elo, host) in team_ids.items():
        _emit_history(conn, rng, tid, elo, NOW)

    venues = [("MetLife Stadium", "East Rutherford"), ("SoFi Stadium", "Los Angeles"),
              ("AT&T Stadium", "Dallas"), ("Estadio Azteca", "Mexico City"),
              ("BC Place", "Vancouver"), ("Mercedes-Benz Stadium", "Atlanta")]

    # Build a set of fixtures: first a played "group round" (settled), then upcoming.
    names = list(team_ids.keys())

    def make_match(home, away, stage, group, kickoff, status, final=False):
        mid = db.new_id()
        th, eh, host_h = team_ids[home]
        ta, ew, _ = team_ids[away]
        venue, city = rng.choice(venues)
        now = db.utc_now()
        hg = ag = None
        if final:
            hg, ag = _final_score(rng, eh, ew, host_h)
        conn.execute(
            """INSERT INTO matches (match_id, fifa_match_id, stage, group_name,
               kickoff_utc, venue_name, city, home_team_id, away_team_id,
               referee_name, weather_summary, rest_days_home, rest_days_away,
               travel_km_home, travel_km_away, status, home_goals, away_goals,
               created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (mid, f"FIFA-{rng.randint(10000,99999)}", stage, group, _iso(kickoff),
             venue, city, th, ta, "M. Oliver", rng.choice(["Clear 24°C", "Cloudy 19°C", "Warm 31°C"]),
             rng.randint(3, 6), rng.randint(3, 6), rng.choice([250.0, 800.0, 1500.0, 3000.0]),
             rng.choice([250.0, 800.0, 1500.0, 3000.0]), status, hg, ag, now, now),
        )
        # standings context (only meaningful for later group matches)
        for tid_, is_home in ((th, True), (ta, False)):
            must_win = 1 if (stage == "Group" and rng.random() < 0.25) else 0
            conn.execute(
                """INSERT INTO standings_context (context_id, match_id, team_id,
                   points_before_match, goal_diff_before_match, qualification_scenario,
                   must_win_flag, draw_acceptable_flag, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (db.new_id(), mid, tid_, rng.randint(0, 6), rng.randint(-3, 3),
                 "Knockout race" if must_win else "On track", must_win,
                 1 if rng.random() < 0.3 else 0, now),
            )
        _emit_odds_for_match(conn, rng, mid, eh, ew, host_h, kickoff, final)
        if final:
            _emit_final_details(conn, rng, mid, th, ta, home, away, hg, ag, kickoff)
        return mid

    # Settled group matches (kickoffs in the days just before "now").
    settled_pairs = [
        ("Argentina", "Japan", "A"), ("France", "Canada", "B"),
        ("Spain", "Morocco", "C"), ("England", "Senegal", "D"),
        ("Brazil", "Mexico", "E"), ("Portugal", "USA", "F"),
        ("Netherlands", "Croatia", "G"), ("Germany", "Uruguay", "H"),
        ("Japan", "Canada", "A"), ("Morocco", "Senegal", "C"),
        ("Mexico", "USA", "F"), ("Croatia", "Uruguay", "G"),
    ]
    for i, (h, a, g) in enumerate(settled_pairs):
        ko = NOW - timedelta(days=2, hours=i * 3)
        make_match(h, a, "Group", g, ko, "Final", final=True)

    # Upcoming matches (next several days).
    upcoming_pairs = [
        ("Argentina", "Morocco", "Group", "A"), ("France", "Japan", "Group", "B"),
        ("Spain", "Senegal", "Group", "C"), ("England", "Croatia", "Group", "D"),
        ("Brazil", "USA", "Group", "E"), ("Portugal", "Mexico", "Group", "F"),
        ("Netherlands", "Germany", "Group", "G"), ("Uruguay", "Canada", "Group", "H"),
        ("Argentina", "France", "Round of 16", None), ("Spain", "Brazil", "Round of 16", None),
        ("England", "Portugal", "Round of 16", None), ("Netherlands", "Morocco", "Round of 16", None),
    ]
    for i, (h, a, stage, g) in enumerate(upcoming_pairs):
        ko = NOW + timedelta(days=1 + i // 2, hours=(i % 2) * 6 + 6)
        make_match(h, a, stage, g, ko, "Scheduled", final=False)

    conn.commit()
    counts = {
        "teams": conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0],
        "matches": conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
        "odds_snapshots": conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0],
        "team_match_stats": conn.execute("SELECT COUNT(*) FROM team_match_stats").fetchone()[0],
    }
    conn.close()
    return counts


if __name__ == "__main__":
    print(seed())
