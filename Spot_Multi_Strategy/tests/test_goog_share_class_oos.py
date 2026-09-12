import unittest

import numpy as np
import pandas as pd

import goog_share_class_oos as strategy
import goog_share_class_oos_report as report


class GoogShareClassTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2014-04-03", "2026-06-30", freq="B", tz="UTC")
        rng = np.random.default_rng(20140403)
        common = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(self.index)))
        spread = 0.003 * np.sin(np.arange(len(self.index)) / 30)
        self.prices = pd.DataFrame({
            "GOOG": common * np.exp(spread / 2),
            "GOOGL": common * np.exp(-spread / 2),
        }, index=self.index)

    def test_executed_weights_are_two_rows_lagged_and_neutral(self):
        weights, diagnostics = strategy.causal_state(self.prices)
        expected = diagnostics["target_state"].shift(2).fillna(0.0)
        pd.testing.assert_series_equal(weights["GOOG"] * 2.0, expected, check_names=False)
        self.assertTrue(np.allclose(weights.sum(axis=1), 0.0))
        self.assertTrue(np.allclose(weights.abs().sum(axis=1).isin([0.0, 1.0]), True))

    def test_future_mutation_does_not_change_past(self):
        baseline, _, _ = strategy.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1, 0] *= 0.8
        rerun, _, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_costs_borrow_and_financing_only_when_active(self):
        _, one, _ = strategy.run_scenario(self.prices, 1.0)
        _, two, _ = strategy.run_scenario(self.prices, 2.0)
        active = one["executed_state"].ne(0.0)
        self.assertTrue((one.loc[~active, "short_borrow"] == 0.0).all())
        self.assertAlmostEqual(float(one.loc[active, "short_borrow"].iloc[0]), 0.005 / 252)
        self.assertAlmostEqual(float(two.loc[active, "financing_cost"].iloc[0]), 0.04 / 252)

    def test_report_has_fixed_scenarios_and_regimes(self):
        results, mc, regimes = report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "calendar_year"})

    def test_short_history_and_gap_fail(self):
        with self.assertRaises(ValueError):
            report.run_experiment(self.prices.loc["2018":], simulations=20)
        gapped = self.prices.drop(self.prices.loc["2020-01-01":"2020-02-01"].index)
        with self.assertRaises(ValueError):
            strategy.validate_prices(gapped)


if __name__ == "__main__":
    unittest.main()
