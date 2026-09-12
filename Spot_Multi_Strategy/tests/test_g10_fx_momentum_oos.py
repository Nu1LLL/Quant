import unittest

import numpy as np
import pandas as pd

import g10_fx_momentum_oos as strategy
import g10_fx_momentum_oos_report as report


class G10FxMomentumTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2007-01-02", "2026-06-30", freq="B", tz="UTC")
        rng = np.random.default_rng(20070131)
        self.prices = pd.DataFrame({
            symbol: 100 * np.cumprod(1 + rng.normal(0, 0.005, len(self.index)))
            for symbol in strategy.SYMBOLS
        }, index=self.index)

    def test_weights_are_lagged_and_dollar_neutral(self):
        weights, _ = strategy.lagged_monthly_weights(self.prices)
        valid = weights.dropna().iloc[0]
        self.assertAlmostEqual(float(valid.sum()), 0.0)
        self.assertAlmostEqual(float(valid.abs().sum()), 1.0)

    def test_future_mutation_does_not_change_past(self):
        baseline, _, _, _ = strategy.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1, 0] *= 0.5
        rerun, _, _, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_fixed_costs_borrow_and_financing(self):
        _, one, _, _ = strategy.run_scenario(self.prices, 1.0)
        _, two, _, _ = strategy.run_scenario(self.prices, 2.0)
        self.assertAlmostEqual(float(one["short_borrow"].iloc[0]), 0.005 / 252)
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)
        self.assertGreater(float(one["trading_cost"].iloc[0]), 0.0)

    def test_report_has_fixed_scenarios_and_regimes(self):
        results, mc, regimes = report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(
            set(regimes["family"]), {"period", "usd_regime", "calendar_year"}
        )

    def test_short_history_fails(self):
        with self.assertRaises(ValueError):
            report.run_experiment(self.prices.loc["2015":], simulations=20)


if __name__ == "__main__":
    unittest.main()
