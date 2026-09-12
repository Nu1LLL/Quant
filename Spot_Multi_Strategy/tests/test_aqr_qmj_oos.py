import unittest

import numpy as np
import pandas as pd

import aqr_qmj_oos as study
import aqr_qmj_oos_report as report


class AqrQmjOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2013-09-30", periods=154, freq="ME")
        rng = np.random.default_rng(83)
        self.raw = pd.Series(0.012 + rng.normal(0, 0.025, len(self.index)), index=self.index)

    def test_cost_and_financing_are_fixed(self):
        one = study.apply_scenario(self.raw, 1.0)
        two = study.apply_scenario(self.raw, 2.0)
        self.assertAlmostEqual(one.iloc[0], self.raw.iloc[0] - 0.03 / 12)
        self.assertAlmostEqual(two.iloc[0], 2 * self.raw.iloc[0] - 0.06 / 12 - 0.04 / 12)

    def test_future_mutation_does_not_change_past(self):
        baseline = study.apply_scenario(self.raw, 1.0)
        changed = self.raw.copy()
        changed.iloc[-1] = -0.7
        rerun = study.apply_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_report_has_fixed_three_scenarios(self):
        frame = pd.DataFrame({"QMJ_GLOBAL": self.raw})
        raw, results, _, _ = report.run_experiment(frame, simulations=20)
        self.assertEqual(len(raw), 154)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})

    def test_missing_month_is_rejected(self):
        frame = pd.DataFrame({"QMJ_GLOBAL": self.raw.drop(self.index[20])})
        with self.assertRaises(ValueError):
            report.run_experiment(frame, simulations=20)


if __name__ == "__main__":
    unittest.main()
