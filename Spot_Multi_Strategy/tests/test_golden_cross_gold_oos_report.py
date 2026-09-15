import unittest

import numpy as np
import pandas as pd

import golden_cross_gold_oos_report as report


class RunExperimentTests(unittest.TestCase):
    def _sixteen_year_prices(self, seed=257, drift=0.0002, scale=0.009):
        dates = pd.bdate_range("2004-11-18", periods=252 * 16, tz="UTC")
        rng = np.random.default_rng(seed)
        prices = 40.0 * np.cumprod(1.0 + rng.normal(drift, scale, len(dates)))
        return pd.Series(prices, index=dates)

    def test_rejects_history_shorter_than_fifteen_years(self):
        dates = pd.bdate_range("2004-11-18", periods=252 * 10, tz="UTC")
        rng = np.random.default_rng(3)
        prices = pd.Series(
            40.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.009, len(dates))),
            index=dates
        )
        with self.assertRaises(ValueError):
            report.run_experiment(prices, simulations=200)

    def test_produces_three_leverage_scenarios_with_fifteen_year_gate(self):
        prices = self._sixteen_year_prices()
        results, monte_carlo, regimes = report.run_experiment(prices, simulations=200)
        self.assertEqual(set(results.keys()), {1.0, 2.0, 3.0})
        for leverage, result in results.items():
            checks = result["validation"]["checks"]
            self.assertIn("sample_at_least_15y", checks)
            self.assertIn("walk_forward_passed", checks)
        self.assertIn("joint_target_probability", monte_carlo)
        self.assertIn("family", regimes.columns)
        self.assertTrue({"period", "event"}.issubset(set(regimes["family"])))


if __name__ == "__main__":
    unittest.main()
