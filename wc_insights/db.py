"""SQLite schema + connection helpers.

The schema mirrors the technical spec's PostgreSQL design. We keep feature
snapshots and odds snapshots separate from final model outputs so the system
stays auditable and back-testable. UUIDs are stored as TEXT, timestamps as
ISO-8601 UTC strings.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "wc_insights.db")


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id        TEXT PRIMARY KEY,
    team_name      TEXT NOT NULL UNIQUE,
    confederation  TEXT,
    fifa_rank      INTEGER,
    elo_rating     REAL,
    squad_value    REAL,
    coach_name     TEXT,
    host_flag      INTEGER DEFAULT 0,
    created_at     TEXT,
    updated_at     TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    fifa_match_id   TEXT,
    stage           TEXT,
    group_name      TEXT,
    kickoff_utc     TEXT NOT NULL,
    venue_name      TEXT,
    city            TEXT,
    home_team_id    TEXT NOT NULL REFERENCES teams(team_id),
    away_team_id    TEXT NOT NULL REFERENCES teams(team_id),
    referee_name    TEXT,
    weather_summary TEXT,
    rest_days_home  INTEGER,
    rest_days_away  INTEGER,
    travel_km_home  REAL,
    travel_km_away  REAL,
    status          TEXT DEFAULT 'Scheduled',
    home_goals      INTEGER,
    away_goals      INTEGER,
    created_at      TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS team_match_stats (
    stat_id             TEXT PRIMARY KEY,
    match_id            TEXT REFERENCES matches(match_id),
    team_id             TEXT REFERENCES teams(team_id),
    opponent_team_id    TEXT REFERENCES teams(team_id),
    venue_type          TEXT,
    result              TEXT,
    goals_for           INTEGER,
    goals_against       INTEGER,
    xg_for              REAL,
    xg_against          REAL,
    shots               INTEGER,
    shots_on_target     INTEGER,
    big_chances_for     INTEGER,
    big_chances_against INTEGER,
    possession_pct      REAL,
    cards_yellow        INTEGER,
    cards_red           INTEGER,
    corners             INTEGER,
    fouls               INTEGER,
    offsides            INTEGER,
    saves               INTEGER,
    xg_source           TEXT,
    source              TEXT,
    match_date          TEXT,
    created_at          TEXT
);

CREATE TABLE IF NOT EXISTS standings_context (
    context_id              TEXT PRIMARY KEY,
    match_id                TEXT REFERENCES matches(match_id),
    team_id                 TEXT REFERENCES teams(team_id),
    points_before_match     INTEGER,
    goal_diff_before_match  INTEGER,
    qualification_scenario  TEXT,
    must_win_flag           INTEGER DEFAULT 0,
    draw_acceptable_flag    INTEGER DEFAULT 0,
    created_at              TEXT
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    odds_snapshot_id    TEXT PRIMARY KEY,
    match_id            TEXT REFERENCES matches(match_id),
    bookmaker           TEXT,
    market_type         TEXT,
    selection           TEXT,
    line_value          REAL,
    decimal_odds        REAL,
    implied_prob_raw    REAL,
    implied_prob_novig  REAL,
    captured_at         TEXT,
    is_opening          INTEGER DEFAULT 0,
    is_closing          INTEGER DEFAULT 0,
    source              TEXT,
    created_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_snapshots(match_id, market_type, selection, captured_at);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    feature_snapshot_id  TEXT PRIMARY KEY,
    match_id             TEXT REFERENCES matches(match_id),
    model_version        TEXT,
    feature_name         TEXT,
    feature_value        REAL,
    feature_generated_at TEXT,
    feature_group        TEXT
);
CREATE INDEX IF NOT EXISTS idx_feat_match ON feature_snapshots(match_id, model_version);

CREATE TABLE IF NOT EXISTS model_outputs (
    output_id              TEXT PRIMARY KEY,
    match_id               TEXT REFERENCES matches(match_id),
    model_version          TEXT,
    lambda_home            REAL,
    lambda_away            REAL,
    prob_home_win          REAL,
    prob_draw              REAL,
    prob_away_win          REAL,
    prob_over_15           REAL,
    prob_over_25           REAL,
    prob_over_35           REAL,
    prob_btts_yes          REAL,
    fair_odds_home         REAL,
    fair_odds_draw         REAL,
    fair_odds_away         REAL,
    confidence_score       REAL,
    prediction_generated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_out_match ON model_outputs(match_id, model_version);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id     TEXT PRIMARY KEY,
    match_id              TEXT REFERENCES matches(match_id),
    model_version         TEXT,
    market_type           TEXT,
    selection             TEXT,
    line_value            REAL,
    bookmaker             TEXT,
    offered_odds          REAL,
    fair_odds             REAL,
    model_prob            REAL,
    market_prob_novig     REAL,
    edge_pct_points       REAL,
    expected_value_pct    REAL,
    clv_reference_odds    REAL,
    recommendation_status TEXT,
    confidence_band       TEXT,
    stake_fraction        REAL,
    created_at            TEXT
);
CREATE INDEX IF NOT EXISTS idx_rec_match ON recommendations(match_id);

CREATE TABLE IF NOT EXISTS settled_bets (
    settled_bet_id     TEXT PRIMARY KEY,
    recommendation_id  TEXT REFERENCES recommendations(recommendation_id),
    match_id           TEXT REFERENCES matches(match_id),
    result_status      TEXT,
    closing_odds       REAL,
    clv_pct            REAL,
    pnl_units          REAL,
    settled_at         TEXT
);

CREATE TABLE IF NOT EXISTS match_events (
    event_id    TEXT PRIMARY KEY,
    match_id    TEXT REFERENCES matches(match_id),
    team_id     TEXT REFERENCES teams(team_id),
    player_name TEXT,
    minute      TEXT,
    event_type  TEXT,   -- goal | penalty | own_goal | yellow | red
    sort_order  INTEGER,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_match ON match_events(match_id, sort_order);

CREATE TABLE IF NOT EXISTS match_lineups (
    match_id            TEXT PRIMARY KEY REFERENCES matches(match_id),
    injury_count_home   INTEGER DEFAULT 0,
    injury_count_away   INTEGER DEFAULT 0,
    lineup_confirmed    INTEGER DEFAULT 0,
    note                TEXT,
    updated_at          TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id    TEXT PRIMARY KEY,
    entity      TEXT,
    entity_id   TEXT,
    action      TEXT,
    detail      TEXT,
    created_at  TEXT
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def reset_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    for suffix in ("", "-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)
    conn = connect(db_path)
    init_db(conn)
    return conn


def audit(conn: sqlite3.Connection, entity: str, entity_id: str, action: str, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO audit_log (audit_id, entity, entity_id, action, detail, created_at) VALUES (?,?,?,?,?,?)",
        (new_id(), entity, entity_id, action, detail, utc_now()),
    )
