"""Offline, synthetic metrics checks; never fetch formal yields or prices."""

import unittest

from research.tips_real_yield_metrics import MC_PATHS, evaluate
from research.tips_real_yield_sim import run_backtest
from test_tips_real_yield_sim import _fixture


class RealYieldMetricsTests(unittest.TestCase):
    def test_flat_price_costs_fail_frozen_gate(self):
        inputs = _fixture()
        primary = run_backtest(*inputs)
        schedule = dict(primary.capacity_schedule)
        benchmark = run_backtest(*inputs, benchmark=True, capacity_schedule=schedule)
        stress = run_backtest(*inputs, slippage_rate=0.0025,
                              capacity_schedule=schedule)
        latency = run_backtest(*inputs, latency_bars=1,
                               capacity_schedule=schedule)
        report = evaluate(primary, benchmark, inputs[1]["BTCUSDT"], stress, latency)
        self.assertEqual(report["full"]["days"], 2099)
        self.assertEqual(len(report["annual_walk_forward"]), 6)
        self.assertEqual(len(report["monthly"]["returns"]), 69)
        self.assertEqual(report["market_regimes"]["unscored_first_days"]["days"], 30)
        self.assertEqual(report["monte_carlo"]["paths"], MC_PATHS)
        self.assertFalse(report["checks"]["full_window"])
        self.assertFalse(report["all_gate_passed"])


if __name__ == "__main__":
    unittest.main()
