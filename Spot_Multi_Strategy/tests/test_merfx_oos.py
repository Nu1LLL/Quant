import unittest

import numpy as np
import pandas as pd

import merfx_oos
import merfx_oos_report


class MerfxOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("1995-01-03", periods=8060, freq="B", tz="UTC")
        rng = np.random.default_rng(103)
        self.prices = pd.Series(
            10 * np.cumprod(1 + 0.00025 + rng.normal(0, 0.002, len(self.index))),
            index=self.index,
        )

    def test_scenario_costs_and_financing(self):
        _, one = merfx_oos.run_scenario(self.prices, 1.0)
        _, two = merfx_oos.run_scenario(self.prices, 2.0)
        self.assertTrue((one["trading_cost"] >= 0).all())
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = merfx_oos.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.75
        rerun, _ = merfx_oos.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_report_has_fixed_three_scenarios(self):
        results, _, _ = merfx_oos_report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})

    def test_twenty_five_year_requirement_is_enforced(self):
        short = self.prices.loc[self.prices.index >= "2010-01-01"]
        with self.assertRaises(ValueError):
            merfx_oos_report.run_experiment(short, simulations=20)


if __name__ == "__main__":
    unittest.main()
