import unittest

import numpy as np
import pandas as pd

import gold_miners_oos_report as report


class RunExperimentTests(unittest.TestCase):
    def _sixteen_year_prices(self, seed=173, drift_a=0.0003, drift_b=0.0002):
        dates = pd.bdate_range("2006-05-22", periods=252 * 16, tz="UTC")
        rng = np.random.default_rng(seed)
        miners = 40.0 * np.cumprod(1.0 + rng.normal(drift_a, 0.02, len(dates)))
        bullion = 40.0 * np.cumprod(1.0 + rng.normal(drift_b, 0.009, len(dates)))
        return pd.Series(miners, index=dates), pd.Series(bullion, index=dates)

    def test_rejects_history_shorter_than_fifteen_years(self):
        dates = pd.bdate_range("2006-05-22", periods=252 * 10, tz="UTC")
        rng = np.random.default_rng(8)
        miners = pd.Series(
            40.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.02, len(dates))), index=dates
        )
        bullion = pd.Series(
            40.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.009, len(dates))), index=dates
        )
        with self.assertRaises(ValueError):
            report.run_experiment(miners, bullion, simulations=200)

    def test_produces_three_leverage_scenarios_with_fifteen_year_gate(self):
        miners, bullion = self._sixteen_year_prices()
        results, monte_carlo, regimes = report.run_experiment(
            miners, bullion, simulations=200
        )
        self.assertEqual(set(results.keys()), {1.0, 2.0, 3.0})
        for leverage, result in results.items():
            checks = result["validation"]["checks"]
            self.assertIn("sample_at_least_15y", checks)
            self.assertNotIn("sample_at_least_5y", checks)
            self.assertIn("walk_forward_passed", checks)
        self.assertIn("joint_target_probability", monte_carlo)
        self.assertIn("family", regimes.columns)
        self.assertTrue({"period", "event"}.issubset(set(regimes["family"])))

    def test_leverage_scales_reported_cagr_direction(self):
        miners, bullion = self._sixteen_year_prices(
            seed=23, drift_a=0.0008, drift_b=0.0001
        )
        results, _, _ = report.run_experiment(miners, bullion, simulations=200)
        cagr_1x = results[1.0]["validation"]["metrics"]["cagr"]
        cagr_3x = results[3.0]["validation"]["metrics"]["cagr"]
        self.assertNotAlmostEqual(cagr_1x, cagr_3x, places=4)


if __name__ == "__main__":
    unittest.main()
