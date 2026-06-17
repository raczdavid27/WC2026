"""Unit tests for the Poisson/Elo model and calibration metrics."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wc_insights import model, calibration


class TestPoisson(unittest.TestCase):
    def test_pmf_sums_to_one(self):
        total = sum(model.poisson_pmf(k, 1.5) for k in range(40))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_pmf_zero_lambda(self):
        self.assertEqual(model.poisson_pmf(0, 0.0), 1.0)
        self.assertEqual(model.poisson_pmf(1, 0.0), 0.0)

    def test_elo_expectancy_symmetry(self):
        self.assertAlmostEqual(model.elo_expectancy(1800, 1800), 0.5)
        self.assertGreater(model.elo_expectancy(2000, 1800), 0.5)


class TestPredict(unittest.TestCase):
    def test_probs_sum_to_one(self):
        pred = model.predict(model.MatchInputs(elo_home=2000, elo_away=1850))
        s = pred.prob_home_win + pred.prob_draw + pred.prob_away_win
        self.assertAlmostEqual(s, 1.0, places=6)

    def test_stronger_team_favored(self):
        pred = model.predict(model.MatchInputs(elo_home=2100, elo_away=1750))
        self.assertGreater(pred.prob_home_win, pred.prob_away_win)

    def test_over_lines_monotonic(self):
        pred = model.predict(model.MatchInputs(elo_home=1900, elo_away=1900))
        self.assertGreaterEqual(pred.prob_over[1.5], pred.prob_over[2.5])
        self.assertGreaterEqual(pred.prob_over[2.5], pred.prob_over[3.5])

    def test_probabilities_in_range(self):
        pred = model.predict(model.MatchInputs(elo_home=2050, elo_away=1800))
        for p in (pred.prob_home_win, pred.prob_draw, pred.prob_away_win,
                  pred.prob_btts_yes, pred.prob_over[2.5]):
            self.assertTrue(0.0 <= p <= 1.0)

    def test_home_field_helps_home(self):
        base = model.predict(model.MatchInputs(elo_home=1850, elo_away=1850))
        host = model.predict(model.MatchInputs(elo_home=1850, elo_away=1850, home_field=True))
        self.assertGreater(host.prob_home_win, base.prob_home_win)

    def test_confidence_band(self):
        self.assertEqual(model.confidence_band(0.9), "High")
        self.assertEqual(model.confidence_band(0.6), "Medium")
        self.assertEqual(model.confidence_band(0.3), "Low")


class TestCalibration(unittest.TestCase):
    def test_brier_perfect(self):
        self.assertAlmostEqual(calibration.brier_score([(1.0, 1), (0.0, 0)]), 0.0)

    def test_brier_worst(self):
        self.assertAlmostEqual(calibration.brier_score([(0.0, 1), (1.0, 0)]), 1.0)

    def test_log_loss_clamped(self):
        # perfectly wrong but clamped -> large finite number
        ll = calibration.log_loss([(0.0, 1)])
        self.assertTrue(ll > 0 and ll < 1e6)

    def test_reliability_buckets(self):
        pairs = [(0.42, 0), (0.43, 1), (0.78, 1), (0.79, 1)]
        rows = calibration.reliability_buckets(pairs, width=0.05)
        self.assertTrue(all("predicted_mean" in r and "observed_freq" in r for r in rows))
        self.assertEqual(sum(r["count"] for r in rows), 4)

    def test_calibration_error_zero_when_perfect(self):
        # predicted 50% and exactly half occur
        pairs = [(0.5, 1), (0.5, 0)]
        self.assertAlmostEqual(calibration.calibration_error(pairs), 0.0)


if __name__ == "__main__":
    unittest.main()
