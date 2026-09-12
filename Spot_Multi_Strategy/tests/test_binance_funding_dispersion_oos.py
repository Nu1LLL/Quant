import unittest

import numpy as np
import pandas as pd

import binance_funding_dispersion_oos as strategy
import binance_funding_dispersion_oos_report as report


class BinanceFundingDispersionTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-03-01", "2026-04-01", freq="8h", tz="UTC")
        symbols = ("ADAUSDT", "BCHUSDT", "LINKUSDT", "LTCUSDT")
        rng = np.random.default_rng(20200201)
        data = {}
        for offset, symbol in enumerate(symbols):
            data[f"{symbol}__funding"] = (
                0.0001 * np.sin(np.arange(len(self.index)) / (17 + offset))
                + offset * 0.00001
            )
            data[f"{symbol}__mark"] = 20 * np.cumprod(
                1 + rng.normal(0, 0.003, len(self.index))
            )
        self.panel = pd.DataFrame(data, index=self.index)

    def test_coverage_gate_and_missing_interval(self):
        self.assertTrue(strategy.coverage_audit(self.panel)["passed"])
        missing = self.panel.drop(self.index[1000:1003])
        self.assertFalse(strategy.coverage_audit(missing)["passed"])

    def test_weights_are_lagged_and_dollar_neutral(self):
        weights, _ = strategy.lagged_rank_weights(self.panel)
        self.assertTrue(weights.iloc[0].isna().all())
        self.assertAlmostEqual(float(weights.iloc[1].sum()), 0.0)
        self.assertAlmostEqual(float(weights.iloc[1].abs().sum()), 1.0)
        changed = self.panel.copy()
        changed.loc[self.index[1]:, "ADAUSDT__funding"] = 1.0
        changed_weights, _ = strategy.lagged_rank_weights(changed)
        pd.testing.assert_series_equal(weights.iloc[1], changed_weights.iloc[1])

    def test_costs_and_financing_are_fixed(self):
        _, one, _ = strategy.run_notional(self.panel, 1.0)
        _, two, _ = strategy.run_notional(self.panel, 2.0)
        self.assertAlmostEqual(float(one["trading_cost"].iloc[0]), 0.0005)
        self.assertAlmostEqual(float(two["financing_cost"].iloc[0]), 0.04 / 1095)

    def test_future_mutation_does_not_change_past(self):
        baseline, _, _ = strategy.run_notional(self.panel, 1.0)
        changed = self.panel.copy()
        changed.iloc[-1, changed.columns.get_loc("BCHUSDT__mark")] *= 0.5
        rerun, _, _ = strategy.run_notional(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_report_has_fixed_scenarios_and_diagnostics(self):
        results, mc, regimes, audit, _ = report.run_experiment(
            self.panel, simulations=20
        )
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertTrue(audit["passed"])
        self.assertEqual(set(regimes["family"]), {"funding_spread", "calendar_year"})


if __name__ == "__main__":
    unittest.main()
