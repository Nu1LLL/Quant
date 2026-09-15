import unittest

import numpy as np
import pandas as pd

import qgmnx_actual_path
import qgmnx_actual_path_report


class QgmnxActualPathTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2015-09-01", periods=2900, freq="B", tz="UTC")
        rng = np.random.default_rng(901)
        self.prices = pd.Series(
            10 * np.cumprod(1 + 0.0004 + rng.normal(0, 0.006, len(self.index))),
            index=self.index,
        )

    def test_only_entry_exit_costs_and_fixed_financing(self):
        _, one = qgmnx_actual_path.run_scenario(self.prices, 1.0)
        _, two = qgmnx_actual_path.run_scenario(self.prices, 2.0)
        self.assertAlmostEqual(float(one["external_cost"].sum()), 0.002)
        self.assertEqual(int((one["external_cost"] > 0).sum()), 2)
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = qgmnx_actual_path.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.8
        rerun, _ = qgmnx_actual_path.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_independent_first_read_and_execution_are_separate(self):
        returns, detail = qgmnx_actual_path.run_scenario(self.prices, 1.0)
        result = qgmnx_actual_path.evaluate_actual_path(
            returns, bool(detail["solvent"].all()), account_executable=False,
            strategy_continuity=False,
        )
        self.assertTrue(result["checks"]["independent_oos"])
        self.assertFalse(result["checks"]["account_executable_for_10000"])
        self.assertFalse(result["checks"]["same_strategy_for_full_sample"])

    def test_report_has_five_frozen_scenarios(self):
        results, _, _, _ = qgmnx_actual_path_report.run_experiment(
            self.prices, simulations=20, account_executable=True,
            strategy_continuity=True,
        )
        self.assertEqual(set(results), set(qgmnx_actual_path.FROZEN_LEVERAGES))

    def test_coverage_gate_rejects_short_history(self):
        short = self.prices.loc[self.prices.index >= "2017-01-01"]
        with self.assertRaises(ValueError):
            qgmnx_actual_path_report.run_experiment(short, simulations=20)


if __name__ == "__main__":
    unittest.main()
