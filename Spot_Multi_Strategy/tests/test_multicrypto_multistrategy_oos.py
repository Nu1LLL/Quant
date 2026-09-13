import unittest

import numpy as np
import pandas as pd

import multicrypto_multistrategy_oos as strategy
import multicrypto_multistrategy_oos_report as report


class MulticryptoMultistrategyOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-06-01", periods=7000, freq="8h", tz="UTC")
        rng = np.random.default_rng(1031)
        pieces = []
        for number, symbol in enumerate(strategy.SYMBOLS):
            returns = rng.normal(0.00005 + number * 0.000002, 0.015, len(self.index))
            mark = (1 + number) * np.cumprod(1 + returns)
            spot = mark * np.cumprod(1 + rng.normal(0.0, 0.0002, len(self.index)))
            pieces.append(pd.DataFrame({
                f"{symbol}__spot": spot,
                f"{symbol}__mark": mark,
                f"{symbol}__funding": rng.normal(0.00008, 0.0001, len(self.index)),
            }, index=self.index))
        self.panel = pd.concat(pieces, axis=1)

    def test_coverage_and_three_sleeves(self):
        self.assertTrue(strategy.coverage_audit(self.panel)["passed"])
        sleeves, _, _ = strategy.sleeve_positions(self.panel)
        active = sleeves["trend_perp"].fillna(0.0).abs().sum(axis=1).gt(0)
        self.assertTrue(np.allclose(sleeves["trend_perp"].loc[active].abs().sum(axis=1), 0.30))
        self.assertTrue(np.allclose(sleeves["dispersion_perp"].dropna().abs().sum(axis=1), 0.30))
        self.assertTrue(np.allclose(sleeves["dispersion_perp"].dropna().sum(axis=1), 0.0))
        self.assertTrue(np.allclose(sleeves["basis_spot"].dropna().sum(axis=1), 0.40))

    def test_future_mutation_does_not_change_past(self):
        baseline, _, _, _, _, _ = strategy.run_scenario(self.panel, 1.0)
        changed = self.panel.copy()
        changed.iloc[-1, :] *= 1.5
        rerun, _, _, _, _, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_costs_financing_scenarios_and_diagnostics(self):
        _, one, _, _, _, _ = strategy.run_scenario(self.panel, 1.0)
        _, two, _, _, _, _ = strategy.run_scenario(self.panel, 2.0)
        self.assertGreater(one["trading_cost"].sum(), 0.0)
        self.assertGreater(two["financing_cost"].sum(), 0.0)
        results, mc, regimes, _ = report.run_experiment(self.panel, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "event"})
        attribution, correlation = report.sleeve_diagnostics(one)
        self.assertEqual(len(attribution), 3)
        self.assertEqual(correlation.shape, (3, 3))

    def test_bad_coverage_missing_symbol_and_gap(self):
        short = self.panel.iloc[1000:]
        self.assertFalse(strategy.coverage_audit(short)["passed"])
        frames = {
            symbol: pd.DataFrame({
                "funding_time": self.index[:3], "spot": [1.0, 1.1, 1.2],
                "perp_mark": [1.0, 1.1, 1.2], "funding_rate": [0.0, 0.0, 0.0],
            }) for symbol in strategy.SYMBOLS[:-1]
        }
        with self.assertRaises(ValueError):
            strategy.build_common_panel(frames)
        gap = self.panel.drop(self.index[100:103])
        self.assertFalse(strategy.coverage_audit(gap)["passed"])

    def test_bankruptcy_stops_at_zero(self):
        sample = pd.Series([0.1, -1.2, 0.8], index=self.index[:3])
        stopped = strategy.apply_bankruptcy(sample)
        self.assertEqual(stopped.tolist(), [0.1, -1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
