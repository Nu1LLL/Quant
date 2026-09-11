import unittest

import numpy as np
import pandas as pd

import regime_barbell_report


class RegimeBarbellTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(regime_barbell_report.START, regime_barbell_report.END, freq="B")
        n = len(self.index)
        self.growth = pd.Series(0.001 + .001*np.sin(np.arange(n)), index=self.index)
        self.barbell = pd.Series(0.0002 + .0005*np.cos(np.arange(n)), index=self.index)
        self.state = pd.Series((np.arange(n)//21 % 2).astype(float), index=self.index)

    def test_binary_state_selects_exactly_one_sleeve(self):
        returns, positions, turnover = regime_barbell_report.build_switch_returns(
            self.growth, self.barbell, self.state
        )
        self.assertTrue((positions.sum(axis=1) == 1).all())
        self.assertTrue((turnover >= 0).all())
        self.assertEqual(len(returns), len(self.index))

    def test_nonbinary_or_missing_state_rejected(self):
        bad = self.state.copy(); bad.iloc[20] = .5
        with self.assertRaises(ValueError):
            regime_barbell_report.build_switch_returns(self.growth, self.barbell, bad)

    def test_future_mutation_does_not_change_past(self):
        before = regime_barbell_report.build_switch_returns(
            self.growth, self.barbell, self.state
        )[0]
        changed = self.state.copy(); changed.iloc[1500:] = 1-changed.iloc[1500:]
        after = regime_barbell_report.build_switch_returns(
            self.growth, self.barbell, changed
        )[0]
        pd.testing.assert_series_equal(before.iloc[:1500], after.iloc[:1500])

    def test_four_scenarios_are_fixed(self):
        summary, artifacts = regime_barbell_report.run_experiment(
            self.growth, self.barbell, self.state, repetitions=39
        )
        expected = {"SWITCH_1x", "SWITCH_2x", "SWITCH_3x", "SWITCH_RISK_OVERLAY"}
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
