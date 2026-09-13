import unittest

import numpy as np
import pandas as pd

import sector_reversal_oos as strategy
import sector_reversal_oos_report as report


class SectorReversalOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("1998-12-22", periods=7200, freq="B", tz="UTC")
        rng = np.random.default_rng(1222)
        self.prices = pd.DataFrame({
            symbol: 25 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(self.index)))
            for symbol in strategy.SYMBOLS
        }, index=self.index)

    def test_signal_is_dollar_neutral_and_delayed_two_rows(self):
        weights, _, targets = strategy.causal_weights(self.prices)
        pd.testing.assert_frame_equal(weights, targets.shift(2).fillna(0.0))
        active = weights.abs().sum(axis=1).gt(0)
        self.assertTrue(np.allclose(weights.loc[active].sum(axis=1), 0.0))
        self.assertTrue(np.allclose(weights.loc[active].abs().sum(axis=1), 1.0))
        self.assertTrue((weights.loc[active].gt(0).sum(axis=1) == 3).all())
        self.assertTrue((weights.loc[active].lt(0).sum(axis=1) == 3).all())

    def test_targets_only_change_from_wednesday_signal(self):
        _, _, targets = strategy.causal_weights(self.prices.iloc[:40])
        changed = targets.ne(targets.shift()).any(axis=1)
        changed.iloc[0] = False
        self.assertTrue((targets.index[changed].weekday == 2).all())

    def test_future_mutation_does_not_change_past(self):
        baseline, _, _, _, _ = strategy.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1, 0] *= 2.0
        rerun, _, _, _, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_costs_borrow_financing_and_scenarios(self):
        one, one_detail, weights, _, _ = strategy.run_scenario(self.prices, 1.0)
        _, two_detail, _, _, _ = strategy.run_scenario(self.prices, 2.0)
        active = weights.abs().sum(axis=1).gt(0)
        self.assertTrue((one_detail.loc[active, "short_borrow"] > 0).all())
        self.assertTrue((two_detail.loc[active, "financing_cost"] > 0).all())
        results, mc, regimes = report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "event"})
        self.assertEqual(len(one), len(one_detail))

    def test_late_start_and_short_sample_fail(self):
        shifted = self.prices.copy()
        shifted.index = pd.date_range("1999-02-01", periods=len(shifted), freq="B", tz="UTC")
        results, _, _ = report.run_experiment(shifted, simulations=20)
        self.assertFalse(results[1.0]["validation"]["checks"]["inception_covered"])
        with self.assertRaises(ValueError):
            report.run_experiment(self.prices.iloc[400:], simulations=20)

    def test_bad_panel_and_long_gap_are_rejected(self):
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices.drop(columns=["XLY"]))
        gapped = self.prices.drop(self.prices.index[100:115])
        with self.assertRaises(ValueError):
            strategy.validate_prices(gapped)


if __name__ == "__main__":
    unittest.main()
