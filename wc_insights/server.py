"""stdlib http.server wiring: versioned REST API + static frontend.

Routes
    GET  /                              -> static/index.html
    GET  /static/*                      -> static assets
    GET  /api/v1/matches                -> upcoming match cards
    GET  /api/v1/matches/{id}           -> full match detail
    GET  /api/v1/recommendations        -> active recommendations
    GET  /api/v1/performance            -> aggregate evaluation metrics
    POST /api/v1/admin/ingest/odds      -> ingest odds snapshots
    POST /api/v1/admin/run-predictions  -> feature gen + scoring + recommend
    POST /api/v1/admin/settle           -> settlement + performance refresh

Admin (POST) routes require the X-Admin-Key header to match config admin_key.
"""

import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import db, api, pipeline
from .recommender import load_config

# Single-flight guard so overlapping refreshes can't run concurrently.
_REFRESH_LOCK = threading.Lock()

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
_DB_PATH = db.DEFAULT_DB_PATH

_CONTENT_TYPES = {".html": "text/html", ".js": "application/javascript",
                  ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml"}


def _full_config():
    cfg = {"recommender": load_config(CONFIG_PATH), "admin_key": "demo-admin-key"}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
        cfg["admin_key"] = raw.get("admin_key", cfg["admin_key"])
    return cfg


class Handler(BaseHTTPRequestHandler):
    server_version = "WCInsights/1.0"

    # -- helpers ----------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        if not os.path.isfile(path):
            self._send_json({"error": "not found"}, 404)
            return
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _body_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _require_admin(self, cfg):
        if self.headers.get("X-Admin-Key") != cfg["admin_key"]:
            self._send_json({"error": "unauthorized — set X-Admin-Key header"}, 401)
            return False
        return True

    def log_message(self, fmt, *args):  # quieter logging
        pass

    # -- GET --------------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}

        if path == "/" or path == "/index.html":
            self._send_file(os.path.join(STATIC_DIR, "index.html"))
            return
        if path.startswith("/static/"):
            self._send_file(os.path.join(STATIC_DIR, path[len("/static/"):]))
            return

        conn = db.connect(_DB_PATH)
        try:
            if path == "/api/v1/matches":
                self._send_json(api.list_matches(
                    conn, qs.get("stage"), qs.get("date"),
                    qs.get("recommendation_status"), qs.get("market_type")))
            elif path.startswith("/api/v1/matches/"):
                mid = path.rsplit("/", 1)[-1]
                detail = api.match_detail(conn, mid)
                self._send_json(detail or {"error": "match not found"}, 200 if detail else 404)
            elif path == "/api/v1/recommendations":
                self._send_json(api.list_recommendations(
                    conn, qs.get("status"), qs.get("min_edge"), qs.get("min_ev"),
                    qs.get("bookmaker"), qs.get("market_type")))
            elif path == "/api/v1/performance":
                self._send_json(api.performance(conn))
            elif path == "/api/v1/recommendation-stats":
                self._send_json(api.recommendation_stats(conn))
            elif path == "/api/v1/health":
                self._send_json({"status": "ok", "model_version": api.MODEL_VERSION})
            else:
                self._send_json({"error": "not found"}, 404)
        finally:
            conn.close()

    # -- POST -------------------------------------------------------------
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        cfg = _full_config()
        # Local UI convenience: trigger a full data refresh (no shared conn held,
        # so the ingest reset can't collide; single-flight via the lock).
        if path == "/api/v1/refresh":
            serve_refresh(self, cfg)
            return
        if not path.startswith("/api/v1/admin/"):
            self._send_json({"error": "not found"}, 404)
            return
        if not self._require_admin(cfg):
            return

        conn = db.connect(_DB_PATH)
        try:
            if path == "/api/v1/admin/run-predictions":
                self._send_json(pipeline.run_predictions(conn, cfg["recommender"]))
            elif path == "/api/v1/admin/settle":
                self._send_json(pipeline.run_settlement(conn))
            elif path == "/api/v1/admin/ingest/odds":
                body = self._body_json()
                if body is None:
                    self._send_json({"error": "invalid JSON"}, 400)
                    return
                self._send_json(_ingest_odds(conn, body))
            elif path == "/api/v1/admin/lineup":
                body = self._body_json()
                if body is None:
                    self._send_json({"error": "invalid JSON"}, 400)
                    return
                self._send_json(_set_lineup(conn, body, cfg["recommender"]))
            else:
                self._send_json({"error": "not found"}, 404)
        finally:
            conn.close()


def _ingest_odds(conn, body):
    """Validate + insert odds snapshots. Accepts {"snapshots": [...]}.

    Each snapshot: match_id, bookmaker, market_type, selection, decimal_odds,
    optional line_value, captured_at, is_opening, is_closing, source.
    """
    snaps = body.get("snapshots", [])
    inserted, errors = 0, []
    for i, s in enumerate(snaps):
        odds = s.get("decimal_odds")
        if odds is None or odds <= 1.0:
            errors.append({"index": i, "error": "decimal_odds must be > 1.0"})
            continue
        if not conn.execute("SELECT 1 FROM matches WHERE match_id=?", (s.get("match_id"),)).fetchone():
            errors.append({"index": i, "error": "unknown match_id"})
            continue
        conn.execute(
            """INSERT INTO odds_snapshots
               (odds_snapshot_id, match_id, bookmaker, market_type, selection,
                line_value, decimal_odds, implied_prob_raw, implied_prob_novig,
                captured_at, is_opening, is_closing, source, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (db.new_id(), s["match_id"], s.get("bookmaker"), s.get("market_type"),
             s.get("selection"), s.get("line_value"), odds, 1.0 / odds, None,
             s.get("captured_at") or db.utc_now(), int(s.get("is_opening", 0)),
             int(s.get("is_closing", 0)), s.get("source", "api"), db.utc_now()),
        )
        inserted += 1
    conn.commit()
    return {"inserted": inserted, "errors": errors}


def _summarize_refresh(summary: dict) -> dict:
    """Flatten the full_refresh result into UI-friendly counts."""
    ing = summary.get("ingest", {})
    return {
        "matches": ing.get("matches"),
        "final_matches": ing.get("final_matches"),
        "wc_matches_in_ratings": summary.get("ratings", {}).get("wc_matches_included"),
        "scored": summary.get("predict", {}).get("scored"),
        "settled_bets": summary.get("settle", {}).get("settled_bets"),
        "bet_log": summary.get("settle", {}).get("bet_log"),
    }


def serve_refresh(handler, cfg):
    """Run a full refresh under the single-flight lock."""
    if not _REFRESH_LOCK.acquire(blocking=False):
        handler._send_json({"error": "a refresh is already running"}, 409)
        return
    try:
        summary = pipeline.full_refresh(cfg["recommender"], _DB_PATH)
        handler._send_json({"status": "ok", "summary": _summarize_refresh(summary)})
    except Exception as exc:  # surface failures to the UI rather than hanging
        handler._send_json({"error": f"refresh failed: {exc}"}, 500)
    finally:
        _REFRESH_LOCK.release()


def _set_lineup(conn, body, rec_cfg):
    """Upsert lineup/injury info and re-score that match (lineup-sensitive)."""
    mid = body.get("match_id")
    match = conn.execute("SELECT * FROM matches WHERE match_id=?", (mid,)).fetchone()
    if not match:
        return {"error": "unknown match_id"}
    conn.execute(
        """INSERT INTO match_lineups (match_id, injury_count_home, injury_count_away,
           lineup_confirmed, note, updated_at) VALUES (?,?,?,?,?,?)
           ON CONFLICT(match_id) DO UPDATE SET
             injury_count_home=excluded.injury_count_home,
             injury_count_away=excluded.injury_count_away,
             lineup_confirmed=excluded.lineup_confirmed,
             note=excluded.note, updated_at=excluded.updated_at""",
        (mid, int(body.get("injury_count_home", 0)), int(body.get("injury_count_away", 0)),
         int(bool(body.get("lineup_confirmed", False))), body.get("note"), db.utc_now()),
    )
    db.audit(conn, "match", mid, "lineup_update", body.get("note", ""))
    result = pipeline.score_match(conn, dict(match), rec_cfg)
    conn.commit()
    return {"updated": mid, "rescored": result}


def serve(host="127.0.0.1", port=5050, db_path=db.DEFAULT_DB_PATH):
    global _DB_PATH
    _DB_PATH = db_path
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"WC Insights serving on http://{host}:{port}  (DB: {db_path})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()
