import unittest

import numpy as np
import pandas as pd

import adanx_actual_path
import adanx_actual_path_report


class AdanxActualPathTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2009-01-15", periods=4650, freq="B", tz="UTC")
        rng = np.random.default_rng(151)
        self.prices = pd.Series(
            10 * np.cumprod(1 + 0.0003 + rng.normal(0, 0.003, len(self.index))),
            index=self.index,
        )

    def test_only_entry_exit_costs_and_fixed_financing(self):
        _, one = adanx_actual_path.run_scenario(self.prices, 1.0)
        _, two = adanx_actual_path.run_scenario(self.prices, 2.0)
        self.assertAlmostEqual(float(one["external_cost"].sum()), 0.002)
        self.assertEqual(int((one["external_cost"] > 0).sum()), 2)
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = adanx_actual_path.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.8
        rerun, _ = adanx_actual_path.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_independent_first_read_and_execution_are_separate(self):
        returns, detail = adanx_actual_path.run_scenario(self.prices, 1.0)
        result = adanx_actual_path.evaluate_actual_path(
            returns, bool(detail["solvent"].all()), account_executable=False
        )
        self.assertTrue(result["checks"]["independent_oos"])
        self.assertFalse(result["checks"]["account_executable_for_10000"])
        self.assertFalse(result["passed"])

    def test_report_has_five_frozen_scenarios(self):
        results, _, _, _ = adanx_actual_path_report.run_experiment(
            self.prices, simulations=20, account_executable=True
        )
        self.assertEqual(set(results), set(adanx_actual_path.FROZEN_LEVERAGES))

    def test_coverage_gate_rejects_late_history(self):
        late = self.prices.loc[self.prices.index >= "2011-01-01"]
        with self.assertRaises(ValueError):
            adanx_actual_path_report.run_experiment(late, simulations=20)


if __name__ == "__main__":
    unittest.main()
