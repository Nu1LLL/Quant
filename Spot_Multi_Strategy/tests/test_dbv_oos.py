import unittest

import numpy as np
import pandas as pd

import dbv_oos
import dbv_oos_report


class DbvOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2006-01-03", periods=4788, freq="B", tz="UTC")
        rng = np.random.default_rng(101)
        self.prices = pd.Series(
            25 * np.cumprod(1 + 0.00035 + rng.normal(0, 0.003, len(self.index))),
            index=self.index,
        )

    def test_scenario_costs_and_financing(self):
        _, one = dbv_oos.run_scenario(self.prices, 1.0)
        _, two = dbv_oos.run_scenario(self.prices, 2.0)
        self.assertTrue((one["trading_cost"] >= 0).all())
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = dbv_oos.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.75
        rerun, _ = dbv_oos.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_report_has_fixed_three_scenarios(self):
        results, _, _ = dbv_oos_report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})

    def test_fifteen_year_requirement_is_enforced(self):
        short = self.prices.loc[self.prices.index >= "2012-01-01"]
        with self.assertRaises(ValueError):
            dbv_oos_report.run_experiment(short, simulations=20)

    def test_missing_inception_history_is_a_hard_failure(self):
        late_index = pd.date_range("2008-01-02", periods=4100, freq="B", tz="UTC")
        late_prices = pd.Series(
            np.linspace(25.0, 35.0, len(late_index)), index=late_index
        )
        results, _, _ = dbv_oos_report.run_experiment(late_prices, simulations=20)
        self.assertFalse(
            results[1.0]["validation"]["checks"]["complete_lifetime_data"]
        )
        self.assertFalse(results[1.0]["validation"]["passed"])


if __name__ == "__main__":
    unittest.main()
