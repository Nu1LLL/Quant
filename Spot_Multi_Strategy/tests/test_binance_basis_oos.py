import unittest

import numpy as np
import pandas as pd

import binance_basis_oos as basis
import binance_basis_oos_report as report


class BinanceBasisOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-09-01", "2026-09-11", freq="8h", tz="UTC")
        self.frames = {}
        for number, symbol in enumerate(report.SYMBOLS):
            n = len(self.index)
            spot = 100*np.exp(np.linspace(0, 1+number*.1, n))
            self.frames[symbol] = pd.DataFrame({
                "funding_time": self.index,
                "spot": spot,
                "perp_mark": spot*(1.001 + .0001*np.sin(np.arange(n))),
                "funding_rate": 0.0001,
            })

    def test_delta_neutral_components_include_basis_and_funding(self):
        result = basis.build_interval_components(self.frames)
        self.assertIn("BTCUSDT__gross", result)
        self.assertIn("BTCUSDT__hedge_turnover", result)
        self.assertTrue((result.filter(like="__hedge_turnover") >= 0).all().all())

    def test_positive_funding_grows_and_costs_nonnegative(self):
        intervals = basis.build_interval_components(self.frames)
        daily, detail = basis.run_notional(intervals, 1.0)
        self.assertGreater((1+daily).prod(), 1.0)
        self.assertTrue((detail["trading_cost"] >= 0).all())
        self.assertTrue((detail["financing_cost"] >= 0).all())

    def test_future_mutation_does_not_change_past(self):
        before = basis.build_interval_components(self.frames)
        changed = {key: value.copy() for key, value in self.frames.items()}
        changed["SOLUSDT"].loc[changed["SOLUSDT"].index[4000]:, "funding_rate"] = .01
        after = basis.build_interval_components(changed)
        pd.testing.assert_frame_equal(before.iloc[:3999], after.iloc[:3999])

    def test_report_has_fixed_three_scenarios_and_strict_validation(self):
        _, results, mc, regimes = report.run_experiment(
            self.frames, monte_carlo_simulations=50
        )
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertTrue(results[1.0]["validation"]["checks"]["independent_oos"])
        self.assertEqual(mc["simulations"], 50)
        self.assertEqual(set(regimes["family"]), {"funding", "btc_trend"})

    def test_incomplete_oos_grid_is_rejected(self):
        changed = {key: value.copy() for key, value in self.frames.items()}
        missing_time = pd.Timestamp("2022-01-01 08:00:00", tz="UTC")
        changed["SOLUSDT"] = changed["SOLUSDT"][
            changed["SOLUSDT"]["funding_time"] != missing_time
        ]
        with self.assertRaises(ValueError):
            report.run_experiment(changed, monte_carlo_simulations=20)


if __name__ == "__main__":
    unittest.main()
