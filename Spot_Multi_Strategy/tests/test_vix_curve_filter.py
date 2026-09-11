import unittest

import numpy as np
import pandas as pd

import vix_curve_filter
import vix_curve_filter_report


class VixCurveFilterTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2016-09-12", periods=1000, freq="B", tz="UTC")
        self.base = pd.Series(0.0005, index=self.index)
        self.vix = pd.Series(15.0, index=self.index)
        self.vix3m = pd.Series(18.0, index=self.index)

    def test_signal_is_shifted_one_day(self):
        self.vix.iloc[10] = 25.0
        position = vix_curve_filter.build_risk_position(
            self.vix, self.vix3m, self.index
        )
        self.assertEqual(position.iloc[10], 1.0)
        self.assertEqual(position.iloc[11], 0.0)
        self.assertEqual(position.iloc[12], 1.0)

    def test_future_mutation_does_not_change_past(self):
        before = vix_curve_filter.build_risk_position(
            self.vix, self.vix3m, self.index
        )
        changed = self.vix.copy()
        changed.iloc[700:] = 30.0
        after = vix_curve_filter.build_risk_position(
            changed, self.vix3m, self.index
        )
        pd.testing.assert_series_equal(before.iloc[:700], after.iloc[:700])

    def test_missing_curve_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            vix_curve_filter.build_risk_position(
                self.vix.drop(self.index[50]), self.vix3m, self.index
            )

    def test_costs_are_nonnegative_and_scenarios_fixed(self):
        self.vix.iloc[::20] = 25.0
        summary, artifacts = vix_curve_filter_report.evaluate(
            self.base, self.vix, self.vix3m, repetitions=39
        )
        expected = {
            "BASE_BLEND_1x", "CURVE_FILTER_1x", "CURVE_FILTER_2x",
            "CURVE_FILTER_3x", "CURVE_FILTER_RISK_OVERLAY",
        }
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        for name in ("CURVE_FILTER_1x", "CURVE_FILTER_2x", "CURVE_FILTER_3x"):
            self.assertTrue((artifacts[name][0]["filter_trading_cost"] >= 0).all())
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
