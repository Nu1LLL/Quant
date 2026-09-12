import unittest

import numpy as np
import pandas as pd

import cew_oos
import cew_oos_report


class CewOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2009-05-06", periods=4500, freq="B", tz="UTC")
        rng = np.random.default_rng(506)
        self.prices = pd.Series(
            25 * np.cumprod(1 + 0.0002 + rng.normal(0, 0.008, len(self.index))),
            index=self.index,
        )

    def test_cost_and_financing_are_fixed(self):
        _, one = cew_oos.run_scenario(self.prices, 1.0)
        _, two = cew_oos.run_scenario(self.prices, 2.0)
        first_move = abs(self.prices.pct_change(fill_method=None).dropna().iloc[0])
        self.assertAlmostEqual(one["trading_cost"].iloc[0], 0.0005 * (1 + first_move))
        self.assertAlmostEqual(two["financing_cost"].iloc[0], 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = cew_oos.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 1.5
        rerun, _ = cew_oos.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_fixed_scenarios_and_regimes(self):
        results, mc, regimes = cew_oos_report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "event"})

    def test_late_start_is_a_hard_gate_failure(self):
        shifted = self.prices.copy()
        shifted.index = pd.date_range("2009-07-01", periods=len(shifted), freq="B", tz="UTC")
        results, _, _ = cew_oos_report.run_experiment(shifted, simulations=20)
        self.assertFalse(results[1.0]["validation"]["checks"]["inception_covered"])
        self.assertFalse(results[1.0]["validation"]["passed"])

    def test_short_history_is_rejected(self):
        with self.assertRaises(ValueError):
            cew_oos_report.run_experiment(self.prices.iloc[400:], simulations=20)


if __name__ == "__main__":
    unittest.main()
