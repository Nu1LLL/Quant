import unittest

import numpy as np
import pandas as pd

import bdry_oos
import bdry_oos_report


class BdryOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2018-03-22", periods=2100, freq="B", tz="UTC")
        rng = np.random.default_rng(139)
        self.prices = pd.Series(
            20 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.018, len(self.index))),
            index=self.index,
        )

    def test_scenario_uses_fixed_cost_and_financing(self):
        _, one = bdry_oos.run_scenario(self.prices, 1.0)
        _, two = bdry_oos.run_scenario(self.prices, 2.0)
        first_gross_return = abs(self.prices.pct_change(fill_method=None).dropna().iloc[0])
        expected_first_cost = first_gross_return * 0.001 + 0.001
        self.assertAlmostEqual(float(one["trading_cost"].iloc[0]), expected_first_cost)
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = bdry_oos.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.7
        rerun, _ = bdry_oos.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_report_has_fixed_three_scenarios(self):
        results, mc, regimes = bdry_oos_report.run_experiment(
            self.prices, simulations=20
        )
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "calendar_year"})

    def test_late_inception_data_is_a_hard_failure(self):
        late_index = pd.date_range("2018-06-01", periods=2100, freq="B", tz="UTC")
        late_prices = pd.Series(np.linspace(20, 40, len(late_index)), index=late_index)
        results, _, _ = bdry_oos_report.run_experiment(late_prices, simulations=20)
        self.assertFalse(results[1.0]["validation"]["checks"]["inception_covered"])
        self.assertFalse(results[1.0]["validation"]["passed"])

    def test_five_year_requirement_is_enforced(self):
        short = self.prices.loc[self.prices.index >= "2023-01-01"]
        with self.assertRaises(ValueError):
            bdry_oos_report.run_experiment(short, simulations=20)


if __name__ == "__main__":
    unittest.main()
