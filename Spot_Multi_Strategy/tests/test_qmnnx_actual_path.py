import unittest

import numpy as np
import pandas as pd

import qmnnx_actual_path
import qmnnx_actual_path_report


class QmnnxActualPathTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2015-01-02", periods=3050, freq="B", tz="UTC")
        rng = np.random.default_rng(1001)
        self.prices = pd.Series(
            10 * np.cumprod(1 + 0.0003 + rng.normal(0, 0.003, len(self.index))),
            index=self.index,
        )

    def test_fixed_high_leverage_grid_and_costs(self):
        self.assertEqual(
            qmnnx_actual_path.FROZEN_LEVERAGES,
            (1.0, 2.0, 3.0, 5.0, 7.5, 10.0),
        )
        _, one = qmnnx_actual_path.run_scenario(self.prices, 1.0)
        _, ten = qmnnx_actual_path.run_scenario(self.prices, 10.0)
        self.assertAlmostEqual(float(one["external_cost"].sum()), 0.002)
        self.assertAlmostEqual(float(ten["financing_cost"].iloc[0]), 9 * 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = qmnnx_actual_path.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.8
        rerun, _ = qmnnx_actual_path.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_strategy_continuity_is_a_hard_gate(self):
        returns, detail = qmnnx_actual_path.run_scenario(self.prices, 1.0)
        result = qmnnx_actual_path.evaluate_actual_path(
            returns, bool(detail["solvent"].all()), True, False
        )
        self.assertFalse(result["checks"]["same_strategy_for_full_sample"])
        self.assertFalse(result["passed"])

    def test_report_has_frozen_scenarios(self):
        results, _, _, _ = qmnnx_actual_path_report.run_experiment(
            self.prices, simulations=20, account_executable=True,
            strategy_continuity=True,
        )
        self.assertEqual(set(results), set(qmnnx_actual_path.FROZEN_LEVERAGES))

    def test_coverage_gate_rejects_short_history(self):
        short = self.prices.loc[self.prices.index >= "2017-01-01"]
        with self.assertRaises(ValueError):
            qmnnx_actual_path_report.run_experiment(short, simulations=20)


if __name__ == "__main__":
    unittest.main()
