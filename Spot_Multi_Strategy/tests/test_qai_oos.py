import unittest

import numpy as np
import pandas as pd

import qai_oos
import qai_oos_report


class QaiOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2010-01-04", periods=3024, freq="B", tz="UTC")
        rng = np.random.default_rng(97)
        self.prices = pd.Series(
            25 * np.cumprod(1 + 0.0004 + rng.normal(0, 0.003, len(self.index))),
            index=self.index,
        )

    def test_costs_nonnegative_and_financing_fixed(self):
        _, one = qai_oos.run_scenario(self.prices, 1.0)
        _, two = qai_oos.run_scenario(self.prices, 2.0)
        self.assertTrue((one["trading_cost"] >= 0).all())
        self.assertAlmostEqual(float(one["financing_cost"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_entry_and_exit_costs_are_charged(self):
        _, detail = qai_oos.run_scenario(self.prices, 1.0)
        gross = self.prices.pct_change(fill_method=None).dropna()
        self.assertAlmostEqual(detail["trading_cost"].iloc[0], gross.abs().iloc[0] * 0.0005 + 0.0005)
        self.assertAlmostEqual(detail["trading_cost"].iloc[-1], gross.abs().iloc[-1] * 0.0005 + 0.0005)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = qai_oos.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.8
        rerun, _ = qai_oos.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_long_gap_is_rejected(self):
        short = self.prices.drop(self.prices.loc["2015-01-01":"2015-02-01"].index)
        with self.assertRaises(ValueError):
            qai_oos.validate_prices(short)

    def test_report_has_fixed_three_scenarios(self):
        results, _, _ = qai_oos_report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})


if __name__ == "__main__":
    unittest.main()
