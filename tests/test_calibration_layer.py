"""Unit tests for the calibration layer (isotonic recalibration) and names."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wc_insights import calibration as cal
from wc_insights.names import norm


class TestPava(unittest.TestCase):
    def test_monotone_output(self):
        xs = [0.1, 0.2, 0.3, 0.4, 0.5]
        ys = [0, 1, 0, 1, 1]  # non-monotone raw
        _, vals = cal._pava(xs, ys)
        self.assertEqual(vals, sorted(vals))  # pooled values non-decreasing


class TestIsotonic(unittest.TestCase):
    def _miscalibrated(self):
        # model says 0.5 but events happen 80% of the time -> needs upward calib
        pairs = [(0.5, 1)] * 80 + [(0.5, 0)] * 20
        return pairs

    def test_fit_predict_monotone(self):
        c = cal.IsotonicCalibrator.fit([(i / 100, 1 if i > 50 else 0) for i in range(100)])
        self.assertLessEqual(c.predict(0.2), c.predict(0.8))

    def test_recovers_frequency(self):
        c = cal.IsotonicCalibrator.fit(self._miscalibrated())
        self.assertAlmostEqual(c.predict(0.5), 0.8, places=2)

    def test_clamps_outside_range(self):
        c = cal.IsotonicCalibrator.fit([(0.3, 0), (0.7, 1)] * 20)
        self.assertTrue(0.0 <= c.predict(0.0) <= 1.0)
        self.assertTrue(0.0 <= c.predict(1.0) <= 1.0)


class TestMarketCalibrators(unittest.TestCase):
    def _oos(self):
        import random
        rng = random.Random(1)
        pairs = []
        for _ in range(400):
            p = rng.random()
            y = 1 if rng.random() < p else 0
            pairs.append((p, y))
        return {"1x2": pairs, "ou25": pairs, "btts": pairs}

    def test_fit_and_apply_renormalizes_1x2(self):
        mc = cal.MarketCalibrators.fit_from_oos(self._oos())
        mk = {"prob_home_win": 0.6, "prob_draw": 0.25, "prob_away_win": 0.15,
              "prob_over": {1.5: 0.8, 2.5: 0.5, 3.5: 0.25}, "prob_btts_yes": 0.5}
        out = mc.calibrate_markets(mk)
        s = out["prob_home_win"] + out["prob_draw"] + out["prob_away_win"]
        self.assertAlmostEqual(s, 1.0, places=6)

    def test_metrics_recorded(self):
        mc = cal.MarketCalibrators.fit_from_oos(self._oos())
        self.assertIn("log_loss_before", mc.metrics["1x2"])
        self.assertIn("ece_after", mc.metrics["1x2"])


class TestNames(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(norm("USA"), "united states")
        self.assertEqual(norm("Czechia"), "czech republic")
        self.assertEqual(norm("Korea Republic"), "south korea")
        self.assertEqual(norm("Bosnia-Herzegovina"), "bosnia")

    def test_accent_and_punct_stripping(self):
        self.assertEqual(norm("Türkiye"), "turkey")
        self.assertEqual(norm("Côte d'Ivoire"), "ivory coast")

    def test_empty(self):
        self.assertEqual(norm(None), "")
        self.assertEqual(norm(""), "")


if __name__ == "__main__":
    unittest.main()
