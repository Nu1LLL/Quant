import unittest

import numpy as np
import pandas as pd

import semiconductor_trend_oos as strategy
import semiconductor_trend_oos_report as report


class SemiconductorTrendOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2010-03-11", periods=4300, freq="B", tz="UTC")
        rng = np.random.default_rng(311)
        soxx = 50 * np.cumprod(1 + rng.normal(0.0004, 0.012, len(self.index)))
        soxl = 20 * np.cumprod(1 + rng.normal(0.0010, 0.035, len(self.index)))
        self.prices = pd.DataFrame({"SOXX": soxx, "SOXL": soxl}, index=self.index)

    def test_month_end_target_is_delayed_two_rows(self):
        weights, _, targets = strategy.causal_weights(self.prices)
        pd.testing.assert_frame_equal(weights, targets.shift(2).fillna(0.0))
        self.assertTrue(weights.abs().sum(axis=1).isin([0.0, 1.0]).all())
        self.assertTrue((weights.gt(0).sum(axis=1) <= 1).all())
        self.assertTrue((weights.lt(0).sum(axis=1) <= 1).all())

    def test_future_price_mutation_does_not_change_past(self):
        baseline, _, _, _ = strategy.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1, :] *= [0.5, 1.5]
        rerun, _, _, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_bankruptcy_stops_equity_at_zero(self):
        sample = pd.Series([0.1, -1.2, 0.8], index=self.index[:3])
        result = strategy.apply_bankruptcy(sample)
        self.assertEqual(result.tolist(), [0.1, -1.0, 0.0])

    def test_costs_borrow_financing_and_scenarios(self):
        _, one, weights, _ = strategy.run_scenario(self.prices, 1.0)
        _, two, _, _ = strategy.run_scenario(self.prices, 2.0)
        short = weights["SOXX"].lt(0)
        active = weights.abs().sum(axis=1).gt(0)
        self.assertTrue((one.loc[short, "short_borrow"] > 0).all())
        self.assertTrue((two.loc[active, "financing_cost"] > 0).all())
        results, mc, regimes = report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "event"})

    def test_late_inception_and_short_span_fail(self):
        shifted = self.prices.copy()
        shifted.index = pd.date_range("2010-05-03", periods=len(shifted), freq="B", tz="UTC")
        results, _, _ = report.run_experiment(shifted, simulations=20)
        self.assertFalse(results[1.0]["validation"]["checks"]["inception_covered"])
        with self.assertRaises(ValueError):
            report.run_experiment(self.prices.iloc[300:], simulations=20)

    def test_bad_columns_and_gap_are_rejected(self):
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices[["SOXX"]])
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices.drop(self.index[100:115]))


if __name__ == "__main__":
    unittest.main()
