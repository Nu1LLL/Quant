import unittest

import numpy as np
import pandas as pd

import qspnx_actual_path
import qspnx_actual_path_report


class QspnxActualPathTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2013-11-01", periods=3360, freq="B", tz="UTC")
        rng = np.random.default_rng(311)
        self.prices = pd.Series(
            10 * np.cumprod(1 + 0.0003 + rng.normal(0, 0.003, len(self.index))),
            index=self.index,
        )

    def test_only_entry_exit_costs_and_fixed_financing(self):
        _, one = qspnx_actual_path.run_scenario(self.prices, 1.0)
        _, two = qspnx_actual_path.run_scenario(self.prices, 2.0)
        self.assertAlmostEqual(float(one["external_cost"].sum()), 0.002)
        self.assertEqual(int((one["external_cost"] > 0).sum()), 2)
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = qspnx_actual_path.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.8
        rerun, _ = qspnx_actual_path.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_independent_oos_is_permanently_false(self):
        returns, detail = qspnx_actual_path.run_scenario(self.prices, 1.0)
        result = qspnx_actual_path.evaluate_actual_path(
            returns, bool(detail["solvent"].all())
        )
        self.assertFalse(result["checks"]["independent_oos"])
        self.assertFalse(result["passed"])

    def test_report_has_five_frozen_scenarios(self):
        results, _, _, _ = qspnx_actual_path_report.run_experiment(
            self.prices, simulations=20
        )
        self.assertEqual(set(results), {1.0, 2.0, 3.0, 4.0, 5.0})

    def test_coverage_gate_rejects_late_history(self):
        late = self.prices.loc[self.prices.index >= "2015-01-01"]
        with self.assertRaises(ValueError):
            qspnx_actual_path_report.run_experiment(late, simulations=20)


if __name__ == "__main__":
    unittest.main()
