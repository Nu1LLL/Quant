import unittest

import numpy as np
import pandas as pd

import iofax_actual_path
import iofax_actual_path_report
import strict_validation


class IofaxActualPathTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2015-06-01", periods=2950, freq="B", tz="UTC")
        rng = np.random.default_rng(528)
        self.prices = pd.Series(
            10 * np.cumprod(1 + 0.00025 + rng.normal(0, 0.003, len(self.index))),
            index=self.index,
        )

    def test_sales_load_and_financing_are_explicit(self):
        _, one = iofax_actual_path.run_scenario(
            self.prices, 1.0, entry_sales_load=0.0575
        )
        _, two = iofax_actual_path.run_scenario(
            self.prices, 2.0, entry_sales_load=0.0575
        )
        self.assertAlmostEqual(float(one["external_cost"].sum()), 0.0595)
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 252)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = iofax_actual_path.run_scenario(self.prices, 1.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 0.8
        rerun, _ = iofax_actual_path.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_nav_quality_rejects_stale_series(self):
        stale = self.prices.copy()
        stale.iloc[1::2] = stale.iloc[:-1:2].to_numpy()
        quality = iofax_actual_path.nav_quality(stale)
        self.assertGreater(quality["zero_return_ratio"], 0.10)
        self.assertFalse(quality["unsmoothed_daily_nav"])

    def test_chunked_monte_carlo_matches_shared_small_case(self):
        returns, _ = iofax_actual_path.run_scenario(self.prices.iloc[:600], 1.0)
        expected = strict_validation.circular_block_monte_carlo(
            returns, simulations=20, block_length=21, seed=77
        )
        actual = iofax_actual_path.circular_block_monte_carlo_chunked(
            returns, simulations=20, block_length=21, seed=77, batch_size=7
        )
        for key in expected:
            self.assertAlmostEqual(expected[key], actual[key])

    def test_report_has_frozen_leverage_grid(self):
        results, _, _, _, _ = iofax_actual_path_report.run_experiment(
            self.prices, simulations=20, account_executable=True,
            strategy_continuity=True, entry_sales_load=0.0575,
        )
        self.assertEqual(set(results), set(iofax_actual_path.FROZEN_LEVERAGES))

    def test_coverage_gate_rejects_short_history(self):
        short = self.prices.loc[self.prices.index >= "2017-01-01"]
        with self.assertRaises(ValueError):
            iofax_actual_path_report.run_experiment(short, simulations=20)


if __name__ == "__main__":
    unittest.main()
