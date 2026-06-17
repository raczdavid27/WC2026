#!/usr/bin/env python3
"""FIFA World Cup Betting Insights — CLI entry point.

Usage:
    python app.py seed            # build the demo database (synthetic data)
    python app.py predict         # run feature gen + scoring + recommendations
    python app.py settle          # settle finished matches + refresh performance
    python app.py refresh         # seed + predict + settle (one-shot demo setup)
    python app.py serve [--port N]# start the web app
    python app.py test            # run the unit-test suite

Quick start:  python app.py refresh  &&  python app.py serve
"""

import argparse
import os
import sys

from wc_insights import db, dixoncoles
from wc_insights.recommender import load_config
from wc_insights import pipeline, seed as seed_mod, live_ingest, fitting, server

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def _cfg():
    return load_config(CONFIG_PATH)


def cmd_seed(_args):
    counts = seed_mod.seed(db.DEFAULT_DB_PATH)
    print("Seeded SYNTHETIC demo database:", counts)


def cmd_ingest(_args):
    print("Ingesting REAL 2026 World Cup data (ESPN + eloratings)…")
    counts = live_ingest.ingest(db.DEFAULT_DB_PATH, reset=True)
    print("Ingested live database:", counts)


def cmd_fit(_args):
    print("Fitting Dixon-Coles ratings + calibration on historical results…")
    report = fitting.fit_all(db_path=db.DEFAULT_DB_PATH)
    pipeline.reload_artifacts()
    print("Fit complete.")
    print("  history:", report["history_matches"], "matches,", report["teams_rated"], "teams rated")
    print("  log-loss:", report["log_loss"])
    print("  DC vs base-rate:", report.get("improvement_vs_base_rate_pct"), "%  | vs indep-Poisson:",
          report.get("improvement_vs_indep_poisson_pct"), "%")


def cmd_predict(_args):
    conn = db.connect(db.DEFAULT_DB_PATH)
    try:
        print("Predictions:", pipeline.run_predictions(conn, _cfg(), only_scheduled=False))
    finally:
        conn.close()


def cmd_settle(_args):
    conn = db.connect(db.DEFAULT_DB_PATH)
    try:
        print("Settlement:", pipeline.run_settlement(conn))
    finally:
        conn.close()


def cmd_capture(_args):
    """Append a fresh odds snapshot (accumulates line history for CLV)."""
    print("Capturing current odds:", live_ingest.capture_odds(db.DEFAULT_DB_PATH))
    cmd_predict(_args)
    cmd_settle(_args)


def cmd_watch(args):
    """Scheduler loop: periodically capture odds, re-score, and settle."""
    import time
    print(f"Watching — capturing odds every {args.interval} min. Ctrl+C to stop.")
    try:
        while True:
            cmd_capture(args)
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        print("\nstopped")


def cmd_update(_args):
    """Refit ratings folding in completed WC results (in-tournament update)."""
    upd = fitting.update_ratings(db.DEFAULT_DB_PATH)
    pipeline.reload_artifacts()
    print("Ratings updated:", upd)


def cmd_refresh(args):
    """Full real-data setup: (fit if needed) -> ingest -> update ratings -> predict -> settle."""
    summary = pipeline.full_refresh(_cfg(), db.DEFAULT_DB_PATH)
    print("Ingested:", summary["ingest"])
    print("Ratings:", summary["ratings"])
    print("Predictions:", summary["predict"]["scored"], "matches scored")
    print("Settlement:", summary["settle"])
    print("Live data ready. Start the app with:  python app.py serve")


def cmd_demo(args):
    """Synthetic-data setup (offline demo): seed -> predict -> settle."""
    cmd_seed(args)
    cmd_predict(args)
    cmd_settle(args)
    print("Demo ready. Start the app with:  python app.py serve")


def cmd_serve(args):
    if not os.path.exists(db.DEFAULT_DB_PATH):
        print("No database found — ingesting live data first...")
        cmd_refresh(args)
    # Honor a PORT env var (used by the preview harness) over the CLI default.
    port = int(os.environ.get("PORT", args.port))
    server.serve(port=port)


def cmd_test(_args):
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(os.path.dirname(__file__), "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def main():
    parser = argparse.ArgumentParser(description="FIFA World Cup Betting Insights")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fit", help="fit Dixon-Coles ratings + calibration on history").set_defaults(func=cmd_fit)
    sub.add_parser("update", help="refit ratings folding in completed WC results").set_defaults(func=cmd_update)
    sub.add_parser("ingest", help="ingest REAL World Cup data (ESPN + eloratings)").set_defaults(func=cmd_ingest)
    sub.add_parser("seed", help="build SYNTHETIC demo data (offline)").set_defaults(func=cmd_seed)
    sub.add_parser("predict").set_defaults(func=cmd_predict)
    sub.add_parser("settle").set_defaults(func=cmd_settle)
    sub.add_parser("refresh", help="live ingest + predict + settle").set_defaults(func=cmd_refresh)
    sub.add_parser("demo", help="synthetic seed + predict + settle").set_defaults(func=cmd_demo)
    sub.add_parser("capture", help="append a current odds snapshot + re-score").set_defaults(func=cmd_capture)
    p_watch = sub.add_parser("watch", help="scheduler loop: capture odds on an interval")
    p_watch.add_argument("--interval", type=int, default=15, help="minutes between captures")
    p_watch.set_defaults(func=cmd_watch)
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--port", type=int, default=5050)
    p_serve.set_defaults(func=cmd_serve)
    sub.add_parser("test").set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
