import unittest

import numpy as np
import pandas as pd

import phdg_oos as strategy
import phdg_oos_report as report


class PhdgOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2012-12-06", periods=3600, freq="B", tz="UTC")
        rng = np.random.default_rng(1206)
        self.prices = pd.Series(
            25 * np.cumprod(1 + rng.normal(0.0003, 0.009, len(self.index))),
            index=self.index, name=strategy.SYMBOL,
        )

    def test_position_is_delayed_two_rows(self):
        _, _, position = strategy.run_scenario(self.prices, 1.0)
        self.assertEqual(position.iloc[0], 0.0)
        self.assertEqual(position.iloc[1], 1.0)
        self.assertTrue(position.iloc[1:].eq(1.0).all())

    def test_future_mutation_does_not_change_past(self):
        baseline, _, _ = strategy.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 2.0
        rerun, _, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_costs_financing_scenarios_and_bankruptcy(self):
        _, one, position = strategy.run_scenario(self.prices, 1.0)
        _, two, _ = strategy.run_scenario(self.prices, 2.0)
        active = position.gt(0)
        self.assertGreater(one["trading_cost"].sum(), 0.0)
        self.assertTrue((two.loc[active, "financing_cost"] > 0).all())
        results, mc, regimes = report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "event"})
        stopped = strategy.apply_bankruptcy(pd.Series([0.1, -1.2, 0.8], index=self.index[:3]))
        self.assertEqual(stopped.tolist(), [0.1, -1.0, 0.0])

    def test_late_start_and_short_span_fail(self):
        shifted = self.prices.copy()
        shifted.index = pd.date_range("2013-02-01", periods=len(shifted), freq="B", tz="UTC")
        results, _, _ = report.run_experiment(shifted, simulations=20)
        self.assertFalse(results[1.0]["validation"]["checks"]["inception_covered"])
        with self.assertRaises(ValueError):
            report.run_experiment(self.prices.iloc[150:], simulations=20)

    def test_bad_values_and_gap_are_rejected(self):
        bad = self.prices.copy()
        bad.iloc[10] = -1.0
        with self.assertRaises(ValueError):
            strategy.validate_prices(bad)
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices.drop(self.index[100:115]))


if __name__ == "__main__":
    unittest.main()
