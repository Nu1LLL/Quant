import unittest

import numpy as np
import pandas as pd

import country_momentum_oos as strategy
import country_momentum_oos_report as report


class CountryMomentumOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("1996-03-18", periods=8000, freq="B", tz="UTC")
        rng = np.random.default_rng(318)
        self.prices = pd.DataFrame({
            symbol: 20 * np.cumprod(1 + rng.normal(0.0002, 0.012, len(self.index)))
            for symbol in strategy.SYMBOLS
        }, index=self.index)

    def test_score_is_fixed_12_1_and_weights_are_delayed(self):
        weights, score, targets = strategy.causal_weights(self.prices)
        expected = self.prices.shift(21).div(self.prices.shift(252)).sub(1.0)
        pd.testing.assert_frame_equal(score, expected, check_freq=False)
        pd.testing.assert_frame_equal(weights, targets.shift(2).fillna(0.0))

    def test_active_weights_are_three_by_three_and_neutral(self):
        weights, _, _ = strategy.causal_weights(self.prices)
        active = weights.abs().sum(axis=1).gt(0)
        self.assertTrue((weights.loc[active].gt(0).sum(axis=1) == 3).all())
        self.assertTrue((weights.loc[active].lt(0).sum(axis=1) == 3).all())
        self.assertTrue(np.allclose(weights.loc[active].sum(axis=1), 0.0))
        self.assertTrue(np.allclose(weights.loc[active].abs().sum(axis=1), 1.0))

    def test_future_mutation_does_not_change_past(self):
        baseline, _, _, _, _ = strategy.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1, 0] *= 2.0
        rerun, _, _, _, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_costs_borrow_financing_and_fixed_scenarios(self):
        _, one, weights, _, _ = strategy.run_scenario(self.prices, 1.0)
        _, two, _, _, _ = strategy.run_scenario(self.prices, 2.0)
        active = weights.abs().sum(axis=1).gt(0)
        self.assertTrue((one.loc[active, "short_borrow"] > 0).all())
        self.assertTrue((two.loc[active, "financing_cost"] > 0).all())
        results, mc, regimes = report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "event"})

    def test_late_start_and_short_sample_fail(self):
        shifted = self.prices.copy()
        shifted.index = pd.date_range("1996-05-01", periods=len(shifted), freq="B", tz="UTC")
        results, _, _ = report.run_experiment(shifted, simulations=20)
        self.assertFalse(results[1.0]["validation"]["checks"]["inception_covered"])
        with self.assertRaises(ValueError):
            report.run_experiment(self.prices.iloc[300:], simulations=20)

    def test_bad_panel_and_gap_are_rejected(self):
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices.drop(columns=["EWP"]))
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices.drop(self.index[100:115]))


if __name__ == "__main__":
    unittest.main()
