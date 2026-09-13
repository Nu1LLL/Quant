import unittest

import numpy as np
import pandas as pd

import dxy_trend_oos as strategy
import dxy_trend_oos_report as report


class DxyTrendOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2000-01-03", periods=6800, freq="B", tz="UTC")
        rng = np.random.default_rng(103)
        self.prices = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.00005, 0.006, len(self.index))),
            index=self.index, name=strategy.SYMBOL,
        )

    def test_month_end_target_is_delayed_two_rows(self):
        positions, _, targets = strategy.causal_positions(self.prices)
        pd.testing.assert_series_equal(positions, targets.shift(2).fillna(0.0))
        self.assertTrue(positions.isin([-1.0, 0.0, 1.0]).all())

    def test_future_mutation_does_not_change_past(self):
        baseline, _, _, _ = strategy.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 1.5
        rerun, _, _, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_costs_roll_financing_and_scenarios(self):
        _, one, positions, _ = strategy.run_scenario(self.prices, 1.0)
        _, two, _, _ = strategy.run_scenario(self.prices, 2.0)
        active = positions.ne(0.0)
        self.assertTrue((one.loc[active, "roll_haircut"] > 0).all())
        self.assertTrue((two.loc[active, "financing_cost"] > 0).all())
        self.assertGreater(one["trading_cost"].sum(), 0.0)
        results, mc, regimes = report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "event"})

    def test_bankruptcy_stops_equity_at_zero(self):
        sample = pd.Series([0.1, -1.2, 0.8], index=self.index[:3])
        result = strategy.apply_bankruptcy(sample)
        self.assertEqual(result.tolist(), [0.1, -1.0, 0.0])
        validation = report.evaluate_oos(result)
        self.assertEqual(validation["metrics"]["cagr"], -1.0)
        self.assertFalse(validation["checks"]["solvent"])

    def test_late_start_and_short_span_fail(self):
        shifted = self.prices.copy()
        shifted.index = pd.date_range("2001-02-01", periods=len(shifted), freq="B", tz="UTC")
        results, _, _ = report.run_experiment(shifted, simulations=20)
        self.assertFalse(results[1.0]["validation"]["checks"]["inception_covered"])
        with self.assertRaises(ValueError):
            report.run_experiment(self.prices.iloc[1700:], simulations=20)

    def test_bad_values_and_gap_are_rejected(self):
        bad = self.prices.copy()
        bad.iloc[10] = -1.0
        with self.assertRaises(ValueError):
            strategy.validate_prices(bad)
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices.drop(self.index[100:115]))


if __name__ == "__main__":
    unittest.main()
