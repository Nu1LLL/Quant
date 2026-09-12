import unittest

import numpy as np
import pandas as pd

import aqr_vme_oos as study
import aqr_vme_oos_report as report


class AqrVmeOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2013-07-31", periods=150, freq="ME")
        rng = np.random.default_rng(67)
        self.frame = pd.DataFrame({
            "VAL": 0.01 + rng.normal(0, 0.03, len(self.index)),
            "MOM": 0.015 + rng.normal(0, 0.025, len(self.index)),
        }, index=self.index)

    def test_equal_weight_is_fixed(self):
        combined = study.equal_value_momentum(self.frame)
        expected = 0.5 * self.frame["VAL"] + 0.5 * self.frame["MOM"]
        pd.testing.assert_series_equal(combined, expected.rename(combined.name))

    def test_cost_and_financing_are_applied(self):
        gross = study.equal_value_momentum(self.frame)
        one = study.apply_scenario(gross, 1.0)
        two = study.apply_scenario(gross, 2.0)
        self.assertAlmostEqual(one.iloc[0], gross.iloc[0] - 0.03 / 12)
        self.assertAlmostEqual(two.iloc[0], 2 * gross.iloc[0] - 0.06 / 12 - 0.04 / 12)

    def test_future_mutation_does_not_change_past(self):
        baseline = study.equal_value_momentum(self.frame)
        changed = self.frame.copy()
        changed.iloc[-1, :] = [-0.8, 0.8]
        rerun = study.equal_value_momentum(changed)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_report_has_fixed_three_scenarios(self):
        _, _, results, _, _, diagnostics = report.run_experiment(
            self.frame, simulations=20
        )
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(diagnostics["oos_months"], 150)


if __name__ == "__main__":
    unittest.main()
