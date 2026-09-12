import unittest

import numpy as np
import pandas as pd

import emlc_oos_report as report


class RunExperimentTests(unittest.TestCase):
    def _sixteen_year_prices(self, seed=41, drift=0.0002, scale=0.006):
        dates = pd.bdate_range("2010-07-23", periods=252 * 16, tz="UTC")
        rng = np.random.default_rng(seed)
        returns = rng.normal(drift, scale, len(dates))
        prices = 20.0 * np.cumprod(1.0 + returns)
        return pd.Series(prices, index=dates)

    def test_rejects_history_shorter_than_fifteen_years(self):
        dates = pd.bdate_range("2010-07-23", periods=252 * 10, tz="UTC")
        rng = np.random.default_rng(1)
        prices = pd.Series(
            20.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.006, len(dates))),
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
            self.assertNotIn("sample_at_least_5y", checks)
            self.assertIn("walk_forward_passed", checks)
        self.assertIn("joint_target_probability", monte_carlo)
        self.assertIn("family", regimes.columns)
        self.assertTrue({"period", "event"}.issubset(set(regimes["family"])))

    def test_leverage_scales_reported_gross_return_direction(self):
        prices = self._sixteen_year_prices(seed=7, drift=0.0006, scale=0.004)
        results, _, _ = report.run_experiment(prices, simulations=200)
        cagr_1x = results[1.0]["validation"]["metrics"]["cagr"]
        cagr_3x = results[3.0]["validation"]["metrics"]["cagr"]
        # 正漂移下，杠杆应该放大名义总收益方向（不对Sharpe或Gate通过
        # 与否做任何断言，那些在真实数据上可能因为成本和波动率拖累
        # 而不是单调的，这里只验证杠杆确实作用在了收益上）
        self.assertNotAlmostEqual(cagr_1x, cagr_3x, places=4)


if __name__ == "__main__":
    unittest.main()
