import unittest

import numpy as np
import pandas as pd

import bitmex_funding
import bitmex_funding_report


class BitmexFundingTests(unittest.TestCase):
    def setUp(self):
        timestamps = pd.date_range(
            "2016-09-11 20:00", periods=12000, freq="8h", tz="UTC"
        )
        rates = np.tile([0.0001, 0.0001, -0.00005, 0.0001], 3000)
        self.funding = pd.DataFrame({
            "timestamp": timestamps,
            "symbol": "XBTUSD",
            "fundingInterval": "PT8H",
            "fundingRate": rates,
        })

    def test_lag_positive_uses_only_previous_rate(self):
        rates = pd.Series([0.1, -0.1, 0.2, 0.0])
        signal = bitmex_funding.build_active_signal(rates, "LAG_POSITIVE")
        self.assertListEqual(signal.tolist(), [0.0, 1.0, 0.0, 1.0])

    def test_always_positive_funding_grows_at_one_x(self):
        positive = self.funding.copy()
        positive["fundingRate"] = 0.0001
        _, daily, _, metrics = bitmex_funding.run_backtest(
            positive, "ALWAYS", 1.0
        )
        self.assertGreater(metrics["cagr"], 0.0)
        self.assertTrue((daily["financing_cost"] == 0.0).all())

    def test_leverage_adds_financing_and_costs_stay_nonnegative(self):
        intervals, daily, _, metrics = bitmex_funding.run_backtest(
            self.funding, "LAG_POSITIVE", 3.0
        )
        self.assertTrue((intervals["trading_cost"] >= 0.0).all())
        self.assertTrue((intervals["financing_cost"] >= 0.0).all())
        self.assertGreater(metrics["total_financing_cost"], 0.0)
        self.assertTrue((daily["cost"] >= 0.0).all())

    def test_experiment_has_eight_fixed_scenarios(self):
        summary, artifacts = bitmex_funding_report.run_experiment(
            self.funding, repetitions=99
        )
        expected = {
            f"{policy}_{leverage}x"
            for policy in ("ALWAYS", "LAG_POSITIVE")
            for leverage in (1, 2, 3, 4)
        }
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
