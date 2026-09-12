import unittest

import numpy as np
import pandas as pd

import aqr_tsmom_oos as study
import aqr_tsmom_oos_report as report


class AqrTsmomOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2013-01-31", periods=156, freq="ME")
        rng = np.random.default_rng(41)
        self.returns = pd.Series(0.02 + rng.normal(0, 0.03, len(self.index)), index=self.index)

    def test_cost_and_financing_scale_are_fixed(self):
        one = study.apply_scenario(self.returns, 1.0)
        two = study.apply_scenario(self.returns, 2.0)
        self.assertAlmostEqual(one.iloc[0], self.returns.iloc[0] - 0.02 / 12)
        self.assertAlmostEqual(two.iloc[0], 2 * self.returns.iloc[0] - 0.04 / 12 - 0.04 / 12)

    def test_walk_forward_requires_ten_complete_years(self):
        folds, passed = study.annual_walk_forward(self.returns)
        self.assertEqual(len(folds), 13)
        self.assertIsInstance(passed, bool)

    def test_future_mutation_does_not_change_past(self):
        baseline = study.apply_scenario(self.returns, 1.0)
        changed = self.returns.copy()
        changed.iloc[-1] = -0.8
        rerun = study.apply_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_monte_carlo_is_reproducible(self):
        a = study.circular_block_monte_carlo(self.returns, simulations=50)
        b = study.circular_block_monte_carlo(self.returns, simulations=50)
        self.assertEqual(a, b)

    def test_report_has_three_fixed_scenarios(self):
        frame = pd.DataFrame({"TSMOM": self.returns})
        results, _, _ = report.run_experiment(frame, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})


if __name__ == "__main__":
    unittest.main()
