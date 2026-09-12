import unittest

import numpy as np
import pandas as pd

import srrix_oos
import srrix_oos_report


class SrrixOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2013-01-03", periods=3400, freq="B", tz="UTC")
        rng = np.random.default_rng(107)
        self.prices = pd.Series(
            10 * np.cumprod(1 + 0.0003 + rng.normal(0, 0.003, len(self.index))),
            index=self.index,
        )

    def test_scenario_costs_and_financing(self):
        _, one = srrix_oos.run_scenario(self.prices, 1.0)
        _, two = srrix_oos.run_scenario(self.prices, 2.0)
        self.assertTrue((one["trading_cost"] >= 0).all())
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = srrix_oos.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.75
        rerun, _ = srrix_oos.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_nav_quality_rejects_stale_series(self):
        stale = pd.Series(np.tile([0.0, 0.0, 0.001], 100), dtype=float)
        quality = srrix_oos.nav_quality(stale)
        self.assertFalse(quality["zero_return_ratio_at_most_10pct"])

    def test_report_has_fixed_three_scenarios(self):
        results, _, _ = srrix_oos_report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        quality = [
            results[leverage]["validation"]["nav_quality"]
            for leverage in (1.0, 2.0, 3.0)
        ]
        self.assertEqual(quality[0], quality[1])
        self.assertEqual(quality[1], quality[2])

    def test_ten_year_requirement_is_enforced(self):
        short = self.prices.loc[self.prices.index >= "2020-01-01"]
        with self.assertRaises(ValueError):
            srrix_oos_report.run_experiment(short, simulations=20)


if __name__ == "__main__":
    unittest.main()
