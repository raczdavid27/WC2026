"""Unit tests for the persistent bet log: grading + log/settle cycle."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wc_insights import betlog, db


class TestGrade(unittest.TestCase):
    def test_1x2(self):
        self.assertEqual(betlog._grade("1X2", "Home", None, 2, 0), "Win")
        self.assertEqual(betlog._grade("1X2", "Home", None, 0, 1), "Loss")
        self.assertEqual(betlog._grade("1X2", "Draw", None, 1, 1), "Win")
        self.assertEqual(betlog._grade("1X2", "Away", None, 0, 2), "Win")

    def test_over_under(self):
        self.assertEqual(betlog._grade("O/U", "Over 2.5", 2.5, 2, 1), "Win")
        self.assertEqual(betlog._grade("O/U", "Under 2.5", 2.5, 1, 1), "Win")
        self.assertEqual(betlog._grade("O/U", "Over 2.5", 2.5, 1, 0), "Loss")

    def test_over_under_void_on_exact_line(self):
        self.assertEqual(betlog._grade("O/U", "Over 2.0", 2.0, 1, 1), "Void")

    def test_btts(self):
        self.assertEqual(betlog._grade("BTTS", "Yes", None, 1, 1), "Win")
        self.assertEqual(betlog._grade("BTTS", "No", None, 1, 1), "Loss")
        self.assertEqual(betlog._grade("BTTS", "No", None, 2, 0), "Win")


class TestLogSettleCycle(unittest.TestCase):
    def setUp(self):
        self.maindb = os.path.join(tempfile.gettempdir(), "wc_main_test.db")
        self.betdb = os.path.join(tempfile.gettempdir(), "wc_bet_test.db")
        for p in (self.maindb, self.betdb):
            for s in ("", "-wal", "-shm"):
                if os.path.exists(p + s):
                    os.remove(p + s)
        self.conn = db.reset_db(self.maindb)
        now = db.utc_now()
        self.conn.execute("INSERT INTO teams (team_id,team_name,created_at,updated_at) VALUES ('h','Home',?,?)", (now, now))
        self.conn.execute("INSERT INTO teams (team_id,team_name,created_at,updated_at) VALUES ('a','Away',?,?)", (now, now))
        self.conn.execute(
            """INSERT INTO matches (match_id,fifa_match_id,stage,kickoff_utc,home_team_id,
               away_team_id,status,created_at,updated_at)
               VALUES ('m1','F1','Group','2026-06-20T12:00:00+00:00','h','a','Scheduled',?,?)""",
            (now, now))
        self.conn.execute(
            """INSERT INTO recommendations (recommendation_id,match_id,market_type,selection,
               bookmaker,offered_odds,model_prob,edge_pct_points,expected_value_pct,
               recommendation_status,confidence_band,created_at)
               VALUES ('r1','m1','1X2','Home','BookA',2.5,0.5,8.0,25.0,'Bet','High',?)""", (now,))
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def test_logs_then_settles_win(self):
        first = betlog.sync(self.conn, self.betdb)
        self.assertEqual(first["newly_logged"], 1)
        self.assertEqual(first["newly_settled"], 0)

        # second sync without a result: still pending, no duplicate log
        again = betlog.sync(self.conn, self.betdb)
        self.assertEqual(again["newly_logged"], 0)

        # match finishes 2-0 -> Home bet wins
        self.conn.execute("UPDATE matches SET status='Final', home_goals=2, away_goals=0 WHERE match_id='m1'")
        self.conn.commit()
        settled = betlog.sync(self.conn, self.betdb)
        self.assertEqual(settled["newly_settled"], 1)

        s = betlog.stats(self.betdb)
        self.assertEqual(s["wins"], 1)
        self.assertEqual(s["settled"], 1)
        self.assertAlmostEqual(s["profit_units"], 1.5)  # 2.5 odds, 1u stake
        self.assertEqual(s["by_status"]["Bet"]["wins"], 1)


if __name__ == "__main__":
    unittest.main()
