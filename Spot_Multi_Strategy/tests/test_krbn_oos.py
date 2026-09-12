import unittest

import numpy as np
import pandas as pd

import krbn_oos
import krbn_oos_report


class KrbnOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-07-31", periods=1600, freq="B", tz="UTC")
        rng = np.random.default_rng(137)
        self.prices = pd.Series(
            25 * np.cumprod(1 + 0.0004 + rng.normal(0, 0.01, len(self.index))),
            index=self.index,
        )

    def test_scenario_costs_and_financing(self):
        _, one = krbn_oos.run_scenario(self.prices, 1.0)
        _, two = krbn_oos.run_scenario(self.prices, 2.0)
        self.assertTrue((one["trading_cost"] >= 0).all())
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = krbn_oos.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.7
        rerun, _ = krbn_oos.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_report_has_fixed_three_scenarios(self):
        results, mc, regimes = krbn_oos_report.run_experiment(
            self.prices, simulations=20
        )
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "calendar_year"})

    def test_late_inception_data_is_a_hard_failure(self):
        late_index = pd.date_range("2020-09-01", periods=1600, freq="B", tz="UTC")
        late_prices = pd.Series(np.linspace(25, 35, len(late_index)), index=late_index)
        results, _, _ = krbn_oos_report.run_experiment(late_prices, simulations=20)
        self.assertFalse(results[1.0]["validation"]["checks"]["inception_covered"])
        self.assertFalse(results[1.0]["validation"]["passed"])

    def test_five_year_requirement_is_enforced(self):
        short = self.prices.loc[self.prices.index >= "2023-01-01"]
        with self.assertRaises(ValueError):
            krbn_oos_report.run_experiment(short, simulations=20)


if __name__ == "__main__":
    unittest.main()
