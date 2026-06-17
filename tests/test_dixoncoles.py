"""Unit tests for the Dixon-Coles model: tau correction, score matrix, markets."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wc_insights import dixoncoles as dc


class TestTau(unittest.TestCase):
    def test_tau_cells(self):
        lam, mu, rho = 1.3, 1.1, -0.1
        self.assertAlmostEqual(dc.tau(0, 0, lam, mu, rho), 1 - lam * mu * rho)
        self.assertAlmostEqual(dc.tau(0, 1, lam, mu, rho), 1 + lam * rho)
        self.assertAlmostEqual(dc.tau(1, 0, lam, mu, rho), 1 + mu * rho)
        self.assertAlmostEqual(dc.tau(1, 1, lam, mu, rho), 1 - rho)

    def test_tau_identity_elsewhere(self):
        self.assertEqual(dc.tau(2, 3, 1.0, 1.0, -0.1), 1.0)
        self.assertEqual(dc.tau(0, 0, 1.0, 1.0, 0.0), 1.0)  # rho=0 -> no correction


class TestScoreMatrix(unittest.TestCase):
    def test_sums_to_one(self):
        M = dc.score_matrix(1.8, 1.2, -0.08)
        self.assertAlmostEqual(M.sum(), 1.0, places=9)

    def test_nonnegative(self):
        M = dc.score_matrix(2.4, 0.5, -0.12)
        self.assertTrue((M >= 0).all())

    def test_tau_shifts_draw_mass(self):
        # negative rho should increase the 0-0 and 1-1 (draw) cells vs independent
        base = dc.score_matrix(1.3, 1.3, 0.0)
        dcm = dc.score_matrix(1.3, 1.3, -0.1)
        self.assertGreater(dcm[0, 0] + dcm[1, 1], base[0, 0] + base[1, 1])


class TestMarkets(unittest.TestCase):
    def test_1x2_sums_to_one(self):
        mk = dc.markets(1.6, 1.1, -0.07)
        s = mk["prob_home_win"] + mk["prob_draw"] + mk["prob_away_win"]
        self.assertAlmostEqual(s, 1.0, places=6)

    def test_over_lines_monotonic(self):
        mk = dc.markets(1.5, 1.5, -0.07)
        o = mk["prob_over"]
        self.assertGreaterEqual(o[1.5], o[2.5])
        self.assertGreaterEqual(o[2.5], o[3.5])

    def test_stronger_attack_favored(self):
        mk = dc.markets(2.5, 0.6, -0.07)
        self.assertGreater(mk["prob_home_win"], mk["prob_away_win"])

    def test_probabilities_in_range(self):
        mk = dc.markets(1.4, 1.7, -0.05)
        for p in (mk["prob_btts_yes"], mk["prob_over"][2.5], mk["prob_draw"]):
            self.assertTrue(0.0 <= p <= 1.0)


class TestLambdas(unittest.TestCase):
    PARAMS = {
        "home_adv": 0.25, "rho": -0.07,
        "attack": {"strong": 0.6, "weak": -0.6},
        "defense": {"strong": 0.5, "weak": -0.5},
    }

    def test_stronger_team_higher_lambda(self):
        lam, mu = dc.lambdas(self.PARAMS, "strong", "weak", neutral=True)
        self.assertGreater(lam, mu)

    def test_home_advantage_applied(self):
        neutral = dc.lambdas(self.PARAMS, "strong", "weak", neutral=True)
        home = dc.lambdas(self.PARAMS, "strong", "weak", neutral=False)
        self.assertGreater(home[0], neutral[0])  # home_adv lifts host lambda

    def test_unrated_team_returns_none(self):
        self.assertIsNone(dc.lambdas(self.PARAMS, "strong", "unknown", neutral=True))


class TestFit(unittest.TestCase):
    def _df(self, importance=False):
        import pandas as pd
        rows = []
        for _ in range(40):
            rows.append({"home": "strong", "away": "weak", "hg": 3, "ag": 0})
            rows.append({"home": "strong", "away": "mid", "hg": 2, "ag": 1})
            rows.append({"home": "mid", "away": "weak", "hg": 1, "ag": 0})
        df = pd.DataFrame(rows)
        df["date"] = "2025-01-01"
        df["neutral"] = True
        df["age_days"] = 100
        if importance:
            df["importance"] = 2.0
        return df

    def test_fit_orders_attack(self):
        p = dc.fit(self._df(), ["strong", "mid", "weak"], iters=300)
        self.assertGreater(p["attack"]["strong"], p["attack"]["weak"])
        self.assertTrue(-0.3 < p["rho"] < 0.3)

    def test_importance_column_accepted(self):
        p = dc.fit(self._df(importance=True), ["strong", "mid", "weak"], iters=200)
        self.assertIn("strong", p["attack"])
        self.assertEqual(set(p["defense"]), {"strong", "mid", "weak"})


if __name__ == "__main__":
    unittest.main()
