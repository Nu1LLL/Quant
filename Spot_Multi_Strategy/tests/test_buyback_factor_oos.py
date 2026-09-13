import unittest

import numpy as np
import pandas as pd

import buyback_factor_oos as strategy
import buyback_factor_oos_report as report


class BuybackFactorOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2006-12-20", periods=5200, freq="B", tz="UTC")
        rng = np.random.default_rng(1220)
        market = rng.normal(0.0003, 0.012, len(self.index))
        self.prices = pd.DataFrame({
            "PKW": 20 * np.cumprod(1 + market + rng.normal(0.0001, 0.003, len(self.index))),
            "IWB": 50 * np.cumprod(1 + market),
        }, index=self.index)

    def test_weights_are_neutral_and_delayed(self):
        _, _, weights = strategy.run_scenario(self.prices, 1.0)
        self.assertTrue((weights.iloc[:1] == 0).all().all())
        active = weights.abs().sum(axis=1).gt(0)
        self.assertTrue(np.allclose(weights.loc[active].sum(axis=1), 0.0))
        self.assertTrue(np.allclose(weights.loc[active].abs().sum(axis=1), 1.0))

    def test_future_mutation_does_not_change_past(self):
        baseline, _, _ = strategy.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1, 0] *= 2.0
        rerun, _, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_costs_borrow_financing_and_scenarios(self):
        _, one, weights = strategy.run_scenario(self.prices, 1.0)
        _, two, _ = strategy.run_scenario(self.prices, 2.0)
        active = weights.abs().sum(axis=1).gt(0)
        self.assertTrue((one.loc[active, "short_borrow"] > 0).all())
        self.assertTrue((two.loc[active, "financing_cost"] > 0).all())
        results, mc, regimes = report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "event"})

    def test_late_start_and_short_span_fail(self):
        shifted = self.prices.copy()
        shifted.index = pd.date_range("2007-02-01", periods=len(shifted), freq="B", tz="UTC")
        results, _, _ = report.run_experiment(shifted, simulations=20)
        self.assertFalse(results[1.0]["validation"]["checks"]["inception_covered"])
        with self.assertRaises(ValueError):
            report.run_experiment(self.prices.iloc[150:], simulations=20)

    def test_bad_panel_and_gap_are_rejected(self):
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices[["PKW"]])
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices.drop(self.index[100:115]))


if __name__ == "__main__":
    unittest.main()
