import unittest

import numpy as np
import pandas as pd

import btc_long_short
import btc_long_short_report


class BtcLongShortTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(
            "2016-08-01", periods=1400, freq="D", tz="UTC"
        )
        self.price = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 1.0, len(self.index))),
            index=self.index,
        )

    def test_positions_are_shifted_and_long_or_short(self):
        positions = btc_long_short.build_positions(self.price)
        self.assertTrue((positions.iloc[:31] == 0.0).all())
        self.assertTrue(set(positions.iloc[31:].unique()).issubset({-1.0, 1.0}))

    def test_future_mutation_does_not_change_past(self):
        before = btc_long_short.build_positions(self.price)
        changed = self.price.copy()
        changed.iloc[900:] *= np.linspace(1.0, 3.0, len(changed) - 900)
        after = btc_long_short.build_positions(changed)
        pd.testing.assert_series_equal(before.iloc[:900], after.iloc[:900])

    def test_short_and_financing_costs_are_nonnegative(self):
        falling = pd.Series(
            100.0 * np.exp(-np.linspace(0.0, 1.0, len(self.index))),
            index=self.index,
        )
        simulation, _, metrics = btc_long_short.run_backtest(
            falling, leverage=2.0,
            start=str(self.index[40].date()), end=str(self.index[-1].date())
        )
        self.assertTrue((simulation["short_cost"] >= 0.0).all())
        self.assertTrue((simulation["financing_cost"] >= 0.0).all())
        self.assertGreater(metrics["total_short_cost"], 0.0)

    def test_experiment_has_seven_fixed_scenarios(self):
        vxth_dates = pd.date_range(
            "2016-09-12", periods=1000, freq="B", tz="UTC"
        )
        vxth = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 0.3, len(vxth_dates))),
            index=vxth_dates,
        )
        summary, artifacts, components = btc_long_short_report.run_experiment(
            self.price, vxth, repetitions=99
        )
        expected = {
            "BTC_LS_1x", "BTC_LS_2x", "BTC_LS_3x",
            "BLEND_1x", "BLEND_2x", "BLEND_3x", "BLEND_RISK_OVERLAY",
        }
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertEqual(
            list(components), ["VXTH", "BTC_TSMOM30_LS"]
        )
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
