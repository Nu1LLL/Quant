import unittest

import numpy as np
import pandas as pd

import strict_validation
import vstg_oos
import vstg_oos_report


class VstgOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2009-01-02", periods=4650, freq="B", tz="UTC")
        rng = np.random.default_rng(616)
        self.levels = pd.Series(
            100 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.008, len(self.index))),
            index=self.index,
        )

    def test_costs_and_financing_are_explicit(self):
        _, one = vstg_oos.run_scenario(self.levels, 1.0)
        _, two = vstg_oos.run_scenario(self.levels, 2.0)
        self.assertAlmostEqual(float(one["implementation_drag"].iloc[0]), 0.02 / 252)
        self.assertAlmostEqual(float(one["external_cost"].sum()), 0.002)
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = vstg_oos.run_scenario(self.levels, 1.0)
        changed = self.levels.copy()
        changed.iloc[-1] *= 0.8
        rerun, _ = vstg_oos.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_chunked_monte_carlo_matches_shared_small_case(self):
        returns, _ = vstg_oos.run_scenario(self.levels.iloc[:600], 1.0)
        expected = strict_validation.circular_block_monte_carlo(
            returns, simulations=20, block_length=21, seed=77
        )
        actual = vstg_oos.circular_block_monte_carlo_chunked(
            returns, simulations=20, block_length=21, seed=77, batch_size=7
        )
        for key in expected:
            self.assertAlmostEqual(expected[key], actual[key])

    def test_report_has_frozen_leverage_grid(self):
        results, _, _, _ = vstg_oos_report.run_experiment(
            self.levels, simulations=20, account_executable=True,
            live_history_at_least_10y=True,
        )
        self.assertEqual(set(results), set(vstg_oos.FROZEN_LEVERAGES))

    def test_coverage_gate_rejects_short_history(self):
        short = self.levels.loc[self.levels.index >= "2013-01-01"]
        with self.assertRaises(ValueError):
            vstg_oos_report.run_experiment(short, simulations=20)

    def test_bankruptcy_is_absorbing(self):
        shocked = self.levels.copy()
        shocked.iloc[-2] = shocked.iloc[-3] * 0.1
        returns, detail = vstg_oos.run_scenario(shocked, 5.0)
        first = np.flatnonzero(returns.to_numpy() == -1.0)[0]
        self.assertTrue((returns.iloc[first + 1:] == 0.0).all())
        self.assertEqual(float(detail["equity"].iloc[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
