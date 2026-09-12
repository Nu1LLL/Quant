import unittest

import numpy as np
import pandas as pd

import btc_buyhold_oos_report as report


class RunExperimentTests(unittest.TestCase):
    def _six_year_prices(self, seed=83, drift=0.0004, scale=0.03):
        dates = pd.date_range("2017-08-17", periods=365 * 6, freq="D", tz="UTC")
        rng = np.random.default_rng(seed)
        returns = rng.normal(drift, scale, len(dates))
        prices = 4000.0 * np.cumprod(1.0 + returns)
        return pd.Series(prices, index=dates)

    def test_rejects_history_shorter_than_five_years(self):
        dates = pd.date_range("2017-08-17", periods=365 * 3, freq="D", tz="UTC")
        rng = np.random.default_rng(1)
        prices = pd.Series(
            4000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.03, len(dates))),
            index=dates
        )
        with self.assertRaises(ValueError):
            report.run_experiment(prices, simulations=200)

    def test_produces_three_leverage_scenarios_with_five_year_gate(self):
        prices = self._six_year_prices()
        results, monte_carlo, regimes = report.run_experiment(prices, simulations=200)
        self.assertEqual(set(results.keys()), {1.0, 2.0, 3.0})
        for leverage, result in results.items():
            checks = result["validation"]["checks"]
            self.assertIn("sample_at_least_5y", checks)
            self.assertIn("walk_forward_passed", checks)
            self.assertFalse(checks["independent_oos"])
        self.assertIn("joint_target_probability", monte_carlo)
        self.assertIn("family", regimes.columns)
        self.assertTrue({"period", "event"}.issubset(set(regimes["family"])))

    def test_uses_crypto_appropriate_cost_not_equity_default(self):
        prices = self._six_year_prices(seed=5)
        results, _, _ = report.run_experiment(prices, simulations=200)
        self.assertAlmostEqual(report.FEE_PLUS_SLIPPAGE, 0.0015, places=6)
        self.assertIn("total_fees", results[1.0]["validation"]["metrics"])


if __name__ == "__main__":
    unittest.main()
