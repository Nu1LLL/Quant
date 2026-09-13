import unittest

import numpy as np
import pandas as pd

import multistrategy_multifx_oos as strategy
import multistrategy_multifx_oos_report as report


class MultistrategyMultifxOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2012-12-06", periods=3600, freq="B", tz="UTC")
        rng = np.random.default_rng(1207)
        common = rng.normal(0.0001, 0.003, len(self.index))
        data = {"PHDG": 25 * np.cumprod(1 + rng.normal(0.0003, 0.009, len(self.index)))}
        for offset, symbol in enumerate(strategy.FX_SYMBOLS):
            data[symbol] = (1 + offset) * np.cumprod(
                1 + common + rng.normal(offset * 0.000005, 0.004 + offset * 0.0002, len(self.index))
            )
        self.prices = pd.DataFrame(data, index=self.index)

    def test_fx_sleeves_are_causal_capped_and_cross_neutral(self):
        trend, cross, _, _, _ = strategy.causal_sleeve_targets(self.prices)
        active_trend = trend.abs().sum(axis=1).gt(0)
        active_cross = cross.abs().sum(axis=1).gt(0)
        self.assertTrue(np.allclose(trend.loc[active_trend].abs().sum(axis=1), 1.0))
        self.assertLessEqual(trend.loc[active_trend].abs().max().max(), 0.3000001)
        self.assertTrue(np.allclose(cross.loc[active_cross].abs().sum(axis=1), 1.0))
        self.assertTrue(np.allclose(cross.loc[active_cross].sum(axis=1), 0.0))
        self.assertTrue((cross.loc[active_cross].gt(0).sum(axis=1) == 2).all())
        self.assertTrue((cross.loc[active_cross].lt(0).sum(axis=1) == 2).all())

    def test_execution_is_delayed_and_future_mutation_is_causal(self):
        baseline, _, positions, _, _, _ = strategy.run_scenario(self.prices, 1.0)
        self.assertEqual(positions[("phdg", "PHDG")].iloc[0], 0.0)
        changed = self.prices.copy()
        changed.iloc[-1] *= 1.5
        rerun, _, _, _, _, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_costs_financing_scenarios_and_diagnostics(self):
        _, one, _, _, _, _ = strategy.run_scenario(self.prices, 1.0)
        _, two, _, _, _, _ = strategy.run_scenario(self.prices, 2.0)
        self.assertGreater(one["trading_cost"].sum(), 0.0)
        self.assertGreater(one["roll_haircut"].sum(), 0.0)
        self.assertGreater(two["financing_cost"].sum(), 0.0)
        results, mc, regimes = report.run_experiment(self.prices, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "event"})
        attribution, correlation = report.sleeve_diagnostics(one)
        self.assertEqual(len(attribution), 3)
        self.assertEqual(correlation.shape, (3, 3))

    def test_bankruptcy_and_sample_gates(self):
        stopped = strategy.apply_bankruptcy(pd.Series([0.1, -1.2, 0.8], index=self.index[:3]))
        self.assertEqual(stopped.tolist(), [0.1, -1.0, 0.0])
        shifted = self.prices.copy()
        shifted.index = pd.date_range("2013-02-01", periods=len(shifted), freq="B", tz="UTC")
        results, _, _ = report.run_experiment(shifted, simulations=20)
        self.assertFalse(results[1.0]["validation"]["checks"]["inception_covered"])
        with self.assertRaises(ValueError):
            report.run_experiment(self.prices.iloc[150:], simulations=20)

    def test_bad_panel_and_gap_are_rejected(self):
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices.drop(columns=["6S=F"]))
        with self.assertRaises(ValueError):
            strategy.validate_prices(self.prices.drop(self.index[100:115]))


if __name__ == "__main__":
    unittest.main()
