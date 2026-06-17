"""Live data ingestion — the REAL 2026 FIFA World Cup.

Replaces the synthetic `seed.py` with actual feeds:

    schedule / teams / venues / stages / scores / match stats
        -> ESPN hidden API (free, no key)        site.api.espn.com .../fifa.world
    team strength (Elo)
        -> eloratings.net  World.tsv + en.teams.tsv
    odds (1X2 moneyline + Over/Under 2.5)
        -> embedded in the same ESPN feed (DraftKings), free.
           Optional upgrade: The Odds API (richer markets) if a key is in config.

Everything writes into the same schema as the synthetic seed, so the rest of
the pipeline (features -> model -> recommendations -> settlement) is unchanged.
"""

import os
from datetime import datetime, timedelta, timezone

import requests

from . import db
from .names import ALIASES, norm  # noqa: F401  (ALIASES re-exported for callers)

LEAGUE = "fifa.world"
SB = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{LEAGUE}/scoreboard"
SUMMARY = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{LEAGUE}/summary"
STANDINGS = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{LEAGUE}/standings"
# The apis/v2 variant exposes the full group structure even before kickoff.
GROUPS = f"https://site.api.espn.com/apis/v2/sports/soccer/{LEAGUE}/standings"
ELO_RATINGS = "https://www.eloratings.net/World.tsv"
ELO_NAMES = "https://www.eloratings.net/en.teams.tsv"

TOURN_START = datetime(2026, 6, 11, tzinfo=timezone.utc)
TOURN_END = datetime(2026, 7, 19, tzinfo=timezone.utc)
HOSTS = {"united states", "canada", "mexico"}
DEFAULT_ELO = 1600.0

# Knockout fixtures carry placeholder competitors ("Group A Winner", "Winner
# Match 73", "Runner-up …") until opponents are decided. We skip these matchups
# — they re-appear automatically on a later refresh once the teams are known.
_PLACEHOLDER_TOKENS = ("winner", "place", "runner", "tbd", "loser")


def is_placeholder(team_block):
    name = (team_block.get("displayName") or team_block.get("name") or "").lower()
    return any(tok in name for tok in _PLACEHOLDER_TOKENS)

STAGE_MAP = {
    "group-stage": "Group", "round-of-32": "Round of 32", "round-of-16": "Round of 16",
    "quarterfinals": "Quarterfinal", "semifinals": "Semifinal",
    "3rd-place": "Third Place", "final": "Final",
}


def american_to_decimal(ml):
    if ml is None:
        return None
    ml = float(ml)
    if ml == 0:
        return None
    return round(1 + (ml / 100.0 if ml > 0 else 100.0 / abs(ml)), 3)


# --------------------------------------------------------------------------
# Elo
# --------------------------------------------------------------------------
def fetch_elo():
    """Return {norm_team_name: elo_rating} from eloratings.net."""
    out = {}
    try:
        names = requests.get(ELO_NAMES, timeout=25)
        names.encoding = "utf-8"
        code_to_name = {}
        for ln in names.text.splitlines():
            cols = ln.split("\t")
            if len(cols) >= 2:
                code_to_name[cols[0]] = cols[1]
        rat = requests.get(ELO_RATINGS, timeout=25)
        rat.encoding = "utf-8"
        for ln in rat.text.splitlines():
            cols = ln.split("\t")
            if len(cols) >= 4 and cols[2] in code_to_name:
                try:
                    out[norm(code_to_name[cols[2]])] = float(cols[3])
                except ValueError:
                    pass
    except requests.RequestException as ex:
        print(f"  [elo] fetch failed ({ex}); using default ratings")
    return out


def _elo_for(elo_map, name):
    n = norm(name)
    if n in elo_map:
        return elo_map[n]
    # tolerant retries on a couple of common variants
    for alt in (n.replace("republic", "").strip(), ALIASES.get(n, n)):
        if alt in elo_map:
            return elo_map[alt]
    return DEFAULT_ELO


# --------------------------------------------------------------------------
# ESPN fetch
# --------------------------------------------------------------------------
def _get(url, params=None):
    try:
        return requests.get(url, params=params or {}, timeout=25).json()
    except (requests.RequestException, ValueError):
        return {}


def _group_lookup():
    """Map norm_team -> group letter from the group-structure endpoint."""
    data = _get(GROUPS)
    lookup = {}
    for grp in data.get("children") or []:
        name = grp.get("name", "")  # e.g. "Group A"
        letter = name.replace("Group", "").strip() or None
        for e in (grp.get("standings") or {}).get("entries", []):
            tn = norm((e.get("team") or {}).get("displayName"))
            if tn:
                lookup[tn] = letter
    return lookup


# --------------------------------------------------------------------------
# Upserts
# --------------------------------------------------------------------------
def _upsert_team(conn, cache, team_block, elo_map):
    name = team_block.get("displayName") or team_block.get("name")
    if not name:
        return None
    key = norm(name)
    if key in cache:
        return cache[key]
    row = conn.execute("SELECT team_id FROM teams WHERE team_name=?", (name,)).fetchone()
    now = db.utc_now()
    if row:
        tid = row["team_id"]
        conn.execute("UPDATE teams SET elo_rating=?, updated_at=? WHERE team_id=?",
                     (_elo_for(elo_map, name), now, tid))
    else:
        tid = db.new_id()
        conn.execute(
            """INSERT INTO teams (team_id, team_name, confederation, fifa_rank,
               elo_rating, squad_value, coach_name, host_flag, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (tid, name, None, None, _elo_for(elo_map, name), None,
             team_block.get("abbreviation"), 1 if key in HOSTS else 0, now, now),
        )
    cache[key] = tid
    return tid


def _odds_snapshots(conn, match_id, kickoff, is_final, odds_block, home_name, away_name):
    """Translate ESPN's embedded DraftKings odds into odds_snapshots rows."""
    if not isinstance(odds_block, dict):
        return
    bookmaker = "DraftKings"
    open_at = (kickoff - timedelta(days=10)).isoformat(timespec="seconds")
    cur_at = (kickoff if is_final else datetime.now(timezone.utc)).isoformat(timespec="seconds")

    def emit(market, selection, line, american, is_opening, is_closing, captured):
        dec = american_to_decimal(american)
        if not dec or dec <= 1.0:
            return
        conn.execute(
            """INSERT INTO odds_snapshots
               (odds_snapshot_id, match_id, bookmaker, market_type, selection,
                line_value, decimal_odds, implied_prob_raw, implied_prob_novig,
                captured_at, is_opening, is_closing, source, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (db.new_id(), match_id, bookmaker, market, selection, line, dec, 1.0 / dec,
             None, captured, is_opening, is_closing, "espn", db.utc_now()),
        )

    ml = odds_block.get("moneyline") or {}
    sel_map = {"Home": ml.get("home"), "Draw": ml.get("draw"), "Away": ml.get("away")}
    for sel, side in sel_map.items():
        if not side:
            continue
        if (side.get("open") or {}).get("odds") is not None:
            emit("1X2", sel, None, side["open"]["odds"], 1, 0, open_at)
        if (side.get("close") or {}).get("odds") is not None:
            emit("1X2", sel, None, side["close"]["odds"], 0, 1 if is_final else 0, cur_at)

    total = odds_block.get("total") or {}
    line = odds_block.get("overUnder")
    for sel, key in (("Over 2.5", "over"), ("Under 2.5", "under")):
        side = total.get(key) or {}
        if (side.get("open") or {}).get("odds") is not None:
            emit("O/U", sel, line, side["open"]["odds"], 1, 0, open_at)
        if (side.get("close") or {}).get("odds") is not None:
            emit("O/U", sel, line, side["close"]["odds"], 0, 1 if is_final else 0, cur_at)


def _to_num(v):
    try:
        return float(str(v).replace("%", ""))
    except (TypeError, ValueError):
        return None


def _ingest_match_stats(conn, event_id, match_id, home_id, away_id, kickoff):
    """Pull per-team played-match stats + goal events from the ESPN summary."""
    summ = _get(SUMMARY, {"event": event_id})
    teams = summ.get("boxscore", {}).get("teams", [])
    if len(teams) < 2:
        return
    _ingest_events(conn, summ, match_id, home_id, away_id)
    # goals from header
    goals = {}
    comp = (summ.get("header", {}).get("competitions") or [{}])[0]
    for c in comp.get("competitors", []):
        goals[norm((c.get("team") or {}).get("displayName"))] = _to_num(c.get("score"))

    for t in teams:
        info = t.get("team", {})
        sm = {s.get("name"): s.get("displayValue") for s in t.get("statistics", [])}
        tn = norm(info.get("displayName") or info.get("name"))
        shots = _to_num(sm.get("totalShots"))
        sot = _to_num(sm.get("shotsOnTarget"))
        gf = goals.get(tn)
        # ESPN gives no xG; use a transparent shot-based proxy.
        xg_for = round(0.09 * (shots or 0) + 0.15 * (sot or 0), 2) if (shots or sot) else None
        team_id = home_id if tn == norm(_team_name(conn, home_id)) else away_id
        opp_id = away_id if team_id == home_id else home_id
        ga = goals.get(norm(_team_name(conn, opp_id)))
        conn.execute(
            """INSERT INTO team_match_stats
               (stat_id, match_id, team_id, opponent_team_id, venue_type, result,
                goals_for, goals_against, xg_for, xg_against, shots, shots_on_target,
                big_chances_for, big_chances_against, possession_pct, cards_yellow,
                cards_red, corners, fouls, offsides, saves, xg_source, source,
                match_date, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (db.new_id(), match_id, team_id, opp_id, "Neutral",
             ("W" if (gf or 0) > (ga or 0) else "D" if (gf or 0) == (ga or 0) else "L"),
             gf, ga, xg_for, None, shots, sot, None, None,
             _to_num(sm.get("possessionPct")), _to_num(sm.get("yellowCards")),
             _to_num(sm.get("redCards")), _to_num(sm.get("wonCorners")),
             _to_num(sm.get("foulsCommitted")), _to_num(sm.get("offsides")),
             _to_num(sm.get("saves")), "proxy", "espn",
             kickoff.isoformat(timespec="seconds"), db.utc_now()),
        )
        # second-pass: fill xg_against from opponent's xg_for after both inserted
    _fill_xg_against(conn, match_id)


def _classify_event(ke):
    """Map an ESPN keyEvent to goal/penalty/own_goal/yellow/red, or None.

    Uses the reliable `scoringPlay` flag for goals (type text varies: "Goal",
    "Goal - Header", "Penalty - Scored", "Own Goal", ...) and keyword matching
    for cards. Missed penalties (scoringPlay False) are ignored.
    """
    ttext = ((ke.get("type") or {}).get("text") or "").lower()
    if ke.get("scoringPlay"):
        if "own goal" in ttext:
            return "own_goal"
        if "penalty" in ttext:
            return "penalty"
        return "goal"
    if "card" in ttext:
        if "red" in ttext or "second yellow" in ttext:
            return "red"
        if "yellow" in ttext:
            return "yellow"
    return None


def _ingest_events(conn, summ, match_id, home_id, away_id):
    """Parse goal scorers + cards from ESPN keyEvents into match_events."""
    name_to_id = {norm(_team_name(conn, home_id)): home_id,
                  norm(_team_name(conn, away_id)): away_id}
    order = 0
    for ke in summ.get("keyEvents", []):
        etype = _classify_event(ke)
        if not etype:
            continue
        team = (ke.get("team") or {}).get("displayName")
        team_id = name_to_id.get(norm(team))
        players = [p.get("athlete", {}).get("displayName") for p in ke.get("participants", [])]
        player = players[0] if players else None
        minute = (ke.get("clock") or {}).get("displayValue")
        conn.execute(
            """INSERT INTO match_events
               (event_id, match_id, team_id, player_name, minute, event_type, sort_order, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (db.new_id(), match_id, team_id, player, minute, etype, order, db.utc_now()),
        )
        order += 1


def _team_name(conn, team_id):
    r = conn.execute("SELECT team_name FROM teams WHERE team_id=?", (team_id,)).fetchone()
    return r["team_name"] if r else ""


def _fill_xg_against(conn, match_id):
    rows = [dict(r) for r in conn.execute(
        "SELECT stat_id, team_id, opponent_team_id, xg_for FROM team_match_stats WHERE match_id=?",
        (match_id,)).fetchall()]
    by_team = {r["team_id"]: r["xg_for"] for r in rows}
    for r in rows:
        opp_xg = by_team.get(r["opponent_team_id"])
        if opp_xg is not None:
            conn.execute("UPDATE team_match_stats SET xg_against=? WHERE stat_id=?",
                         (opp_xg, r["stat_id"]))


# --------------------------------------------------------------------------
# Main ingest
# --------------------------------------------------------------------------
def ingest(db_path: str = db.DEFAULT_DB_PATH, reset: bool = True) -> dict:
    conn = db.reset_db(db_path) if reset else db.connect(db_path)
    if not reset:
        db.init_db(conn)

    print("  fetching Elo ratings…")
    elo_map = fetch_elo()
    groups = _group_lookup()
    team_cache = {}

    matches = 0
    day = TOURN_START
    print("  walking ESPN schedule…")
    while day <= TOURN_END:
        sb = _get(SB, {"dates": day.strftime("%Y%m%d")})
        for ev in sb.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue
            home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            home_t, away_t = home.get("team", {}), away.get("team", {})

            # Skip knockout fixtures with undetermined (placeholder) opponents.
            if is_placeholder(home_t) or is_placeholder(away_t):
                continue

            home_id = _upsert_team(conn, team_cache, home_t, elo_map)
            away_id = _upsert_team(conn, team_cache, away_t, elo_map)
            if not home_id or not away_id:
                continue

            status = ev.get("status", {}).get("type", {}).get("name", "")
            is_final = status in ("STATUS_FULL_TIME", "STATUS_FINAL")
            stage = STAGE_MAP.get((ev.get("season") or {}).get("slug", ""),
                                  (ev.get("season") or {}).get("slug", "Group").title())
            venue = comp.get("venue") or ev.get("venue") or {}
            kickoff = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
            grp = groups.get(norm(home_t.get("displayName"))) if stage == "Group" else None

            hg = _to_num(home.get("score")) if is_final else None
            ag = _to_num(away.get("score")) if is_final else None
            mid = db.new_id()
            now = db.utc_now()
            conn.execute(
                """INSERT INTO matches (match_id, fifa_match_id, stage, group_name,
                   kickoff_utc, venue_name, city, home_team_id, away_team_id,
                   referee_name, weather_summary, rest_days_home, rest_days_away,
                   travel_km_home, travel_km_away, status, home_goals, away_goals,
                   created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (mid, str(ev.get("id")), stage, grp, kickoff.isoformat(timespec="seconds"),
                 venue.get("fullName"), (venue.get("address") or {}).get("city"),
                 home_id, away_id, None, None, None, None, None, None,
                 "Final" if is_final else "Live" if status == "STATUS_IN_PROGRESS" else "Scheduled",
                 int(hg) if hg is not None else None, int(ag) if ag is not None else None,
                 now, now),
            )

            odds_list = [o for o in (comp.get("odds") or []) if isinstance(o, dict)]
            if odds_list:
                _odds_snapshots(conn, mid, kickoff, is_final, odds_list[0],
                                home_t.get("displayName"), away_t.get("displayName"))
            if is_final:
                _ingest_match_stats(conn, ev.get("id"), mid, home_id, away_id, kickoff)
            matches += 1
        day += timedelta(days=1)

    conn.commit()
    counts = {
        "teams": conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0],
        "matches": matches,
        "odds_snapshots": conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0],
        "final_matches": conn.execute("SELECT COUNT(*) FROM matches WHERE status='Final'").fetchone()[0],
        "elo_teams_matched": sum(1 for k in team_cache if k in elo_map),
    }
    conn.close()
    return counts


# --------------------------------------------------------------------------
# Incremental odds capture (for real CLV / line movement over time)
# --------------------------------------------------------------------------
def capture_odds(db_path: str = db.DEFAULT_DB_PATH) -> dict:
    """Append a fresh 'current' odds snapshot to existing matches without
    resetting the DB. Run on a schedule so opening->closing history accumulates
    and CLV becomes meaningful. Pulls ESPN embedded odds; if an Odds API key is
    configured it also captures multi-book prices.
    """
    conn = db.connect(db_path)
    by_fifa = {str(r["fifa_match_id"]): r["match_id"]
               for r in conn.execute("SELECT match_id, fifa_match_id FROM matches").fetchall()}
    inserted = 0
    now = datetime.now(timezone.utc)
    day = TOURN_START
    while day <= TOURN_END:
        sb = _get(SB, {"dates": day.strftime("%Y%m%d")})
        for ev in sb.get("events", []):
            mid = by_fifa.get(str(ev.get("id")))
            if not mid:
                continue
            comp = (ev.get("competitions") or [{}])[0]
            odds_list = [o for o in (comp.get("odds") or []) if isinstance(o, dict)]
            if not odds_list:
                continue
            inserted += _capture_current(conn, mid, odds_list[0], now)
        day += timedelta(days=1)
    # optional multi-book via Odds API
    inserted += _oddsapi_capture(conn, now)
    conn.commit()
    conn.close()
    return {"snapshots_added": inserted, "captured_at": now.isoformat(timespec="seconds")}


def _capture_current(conn, match_id, odds_block, captured):
    """Insert the current ESPN price for each 1X2 / O/U selection (is_opening=0)."""
    if not isinstance(odds_block, dict):
        return 0
    n = 0
    ts = captured.isoformat(timespec="seconds")

    def emit(market, selection, line, american):
        nonlocal n
        dec = american_to_decimal(american)
        if not dec or dec <= 1.0:
            return
        conn.execute(
            """INSERT INTO odds_snapshots (odds_snapshot_id, match_id, bookmaker,
               market_type, selection, line_value, decimal_odds, implied_prob_raw,
               implied_prob_novig, captured_at, is_opening, is_closing, source, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (db.new_id(), match_id, "DraftKings", market, selection, line, dec, 1.0 / dec,
             None, ts, 0, 0, "espn", db.utc_now()))
        n += 1

    ml = odds_block.get("moneyline") or {}
    for sel, key in (("Home", "home"), ("Draw", "draw"), ("Away", "away")):
        side = (ml.get(key) or {}).get("close") or {}
        if side.get("odds") is not None:
            emit("1X2", sel, None, side["odds"])
    total = odds_block.get("total") or {}
    line = odds_block.get("overUnder")
    for sel, key in (("Over 2.5", "over"), ("Under 2.5", "under")):
        side = (total.get(key) or {}).get("close") or {}
        if side.get("odds") is not None:
            emit("O/U", sel, line, side["odds"])
    return n


# --------------------------------------------------------------------------
# Optional multi-book odds via The Odds API (opt-in, line shopping)
# --------------------------------------------------------------------------
def _load_config():
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _oddsapi_config():
    cfg = _load_config()
    return cfg if (cfg.get("odds_api_key") or "") else None


def _oddsapi_capture(conn, captured) -> int:
    """If an Odds API key is configured, capture multi-book 1X2 + O/U prices.

    Enables best-price line shopping across books (the recommender already picks
    the best price per selection). No-op when no key is present.
    """
    cfg = _oddsapi_config()
    if not cfg:
        return 0
    key, region = cfg["odds_api_key"], cfg.get("region", "eu")
    try:
        data = requests.get(
            "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds",
            params={"apiKey": key, "regions": region, "markets": "h2h,totals",
                    "oddsFormat": "decimal", "dateFormat": "iso"}, timeout=30).json()
    except (requests.RequestException, ValueError):
        return 0
    if not isinstance(data, list):
        return 0

    # index existing matches by normalized {home, away}
    idx = {}
    for r in conn.execute("""SELECT m.match_id, th.team_name h, ta.team_name a FROM matches m
                             JOIN teams th ON m.home_team_id=th.team_id
                             JOIN teams ta ON m.away_team_id=ta.team_id""").fetchall():
        idx[(norm(r["h"]), norm(r["a"]))] = r["match_id"]

    ts = captured.isoformat(timespec="seconds")
    n = 0
    for ev in data:
        mid = idx.get((norm(ev.get("home_team")), norm(ev.get("away_team"))))
        if not mid:
            continue
        home, away = ev["home_team"], ev["away_team"]
        for bk in ev.get("bookmakers", []):
            book = bk.get("title", bk.get("key"))
            for mkt in bk.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        sel = ("Home" if o["name"] == home else "Away" if o["name"] == away
                               else "Draw")
                        n += _insert_book_odds(conn, mid, book, "1X2", sel, None, o["price"], ts)
                elif mkt["key"] == "totals":
                    for o in mkt["outcomes"]:
                        pt = o.get("point")
                        if pt is None:
                            continue
                        sel = f"{o['name']} {pt}"  # "Over 2.5" / "Under 2.5"
                        n += _insert_book_odds(conn, mid, book, "O/U", sel, pt, o["price"], ts)
    return n


def _insert_book_odds(conn, mid, book, market, sel, line, dec, ts):
    if not dec or dec <= 1.0:
        return 0
    conn.execute(
        """INSERT INTO odds_snapshots (odds_snapshot_id, match_id, bookmaker, market_type,
           selection, line_value, decimal_odds, implied_prob_raw, implied_prob_novig,
           captured_at, is_opening, is_closing, source, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (db.new_id(), mid, book, market, sel, line, dec, 1.0 / dec, None, ts, 0, 0,
         "oddsapi", db.utc_now()))
    return 1


if __name__ == "__main__":
    print(ingest())
