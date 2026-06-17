"""Unit tests for the value engine — no-vig, EV, fair odds, edge, CLV, staking."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wc_insights import value_engine as ve


class TestImpliedAndNoVig(unittest.TestCase):
    def test_implied_prob_raw(self):
        self.assertAlmostEqual(ve.implied_prob_raw(2.0), 0.5)
        self.assertAlmostEqual(ve.implied_prob_raw(4.0), 0.25)

    def test_implied_prob_rejects_bad_odds(self):
        for bad in (1.0, 0.5, 0.0, -1.0, None):
            with self.assertRaises(ValueError):
                ve.implied_prob_raw(bad)

    def test_overround_positive_for_margin_market(self):
        # a 3-way market priced with margin has overround > 0
        self.assertGreater(ve.overround([2.5, 3.2, 2.8]), 0.0)

    def test_novig_probs_sum_to_one(self):
        probs = ve.novig_probs([2.5, 3.2, 2.8])
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        for p in probs:
            self.assertTrue(0.0 < p < 1.0)

    def test_novig_two_way_symmetry(self):
        # equal odds -> equal no-vig probs
        p = ve.novig_probs([1.90, 1.90])
        self.assertAlmostEqual(p[0], 0.5)
        self.assertAlmostEqual(p[1], 0.5)

    def test_novig_removes_overround(self):
        odds = [2.0, 4.0, 4.0]  # raw implied 0.5+0.25+0.25 = 1.0 (no margin)
        probs = ve.novig_probs(odds)
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertAlmostEqual(probs[1], 0.25)
        self.assertAlmostEqual(probs[2], 0.25)


class TestEdgeEvFair(unittest.TestCase):
    def test_edge(self):
        self.assertAlmostEqual(ve.edge(0.55, 0.50), 0.05)
        self.assertAlmostEqual(ve.edge_pct_points(0.55, 0.50), 5.0)

    def test_expected_value(self):
        # fair coin at 2.10 -> EV = 0.5*2.10 - 1 = +0.05
        self.assertAlmostEqual(ve.expected_value(0.5, 2.10), 0.05)
        self.assertAlmostEqual(ve.expected_value_pct(0.5, 2.10), 5.0)

    def test_ev_zero_at_fair_price(self):
        self.assertAlmostEqual(ve.expected_value(0.5, 2.0), 0.0)

    def test_fair_odds_is_reciprocal(self):
        self.assertAlmostEqual(ve.fair_odds(0.25), 4.0)
        self.assertAlmostEqual(ve.fair_odds(0.5), 2.0)

    def test_fair_odds_rejects_zero(self):
        with self.assertRaises(ValueError):
            ve.fair_odds(0.0)


class TestClv(unittest.TestCase):
    def test_positive_clv_when_beating_close(self):
        # took 2.10, closed at 2.00 -> beat the close
        self.assertAlmostEqual(ve.closing_line_value(2.10, 2.00), 0.05)
        self.assertAlmostEqual(ve.closing_line_value_pct(2.10, 2.00), 5.0)

    def test_negative_clv_when_line_moves_against(self):
        self.assertLess(ve.closing_line_value(1.90, 2.00), 0.0)

    def test_zero_clv_at_same_price(self):
        self.assertAlmostEqual(ve.closing_line_value(2.0, 2.0), 0.0)


class TestStakingAndPnl(unittest.TestCase):
    def test_kelly_positive_edge(self):
        # p=0.55 at 2.10 -> positive quarter-Kelly stake
        f = ve.kelly_fraction(0.55, 2.10, 0.25)
        self.assertGreater(f, 0.0)
        self.assertLess(f, 1.0)

    def test_kelly_no_edge_is_zero(self):
        self.assertEqual(ve.kelly_fraction(0.4, 2.0, 0.25), 0.0)

    def test_pnl_win_loss_void(self):
        self.assertAlmostEqual(ve.pnl_units("Win", 1.0, 2.5), 1.5)
        self.assertAlmostEqual(ve.pnl_units("Loss", 1.0, 2.5), -1.0)
        self.assertAlmostEqual(ve.pnl_units("Void", 1.0, 2.5), 0.0)

    def test_best_odds(self):
        snaps = [{"decimal_odds": 2.0, "bookmaker": "A"},
                 {"decimal_odds": 2.2, "bookmaker": "B"}]
        self.assertEqual(ve.best_odds(snaps)["bookmaker"], "B")


if __name__ == "__main__":
    unittest.main()
