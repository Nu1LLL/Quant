"""Synthetic TGA rule checks; no official balance or market outcome is read."""

import unittest

import pandas as pd

from research.tga_cash_sim import END, _direction, run_backtest
from research.tga_cash_source import TgaObservation
from tests.test_tips_real_yield_sim import _fixture


def _inputs(*, declining: bool):
    old, trade, mark, funding = _fixture()
    observations = tuple(TgaObservation(
        row.observed_date,
        100_000 - i if declining else 100_000 + i,
        row.proxy_available_at) for i, row in enumerate(old))
    return observations, trade, mark, funding


class TgaCashSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.long_inputs = _inputs(declining=True)

    def test_declining_balance_buys_and_cost_stress_reuses_schedule(self):
        baseline = run_backtest(*self.long_inputs)
        schedule = dict(baseline.capacity_schedule)
        self.assertEqual(len(schedule), 300)
        self.assertTrue(all(schedule.values()))
        self.assertEqual(len(baseline.records), 2099)
        self.assertTrue(baseline.terminal_flat)
        self.assertLess(baseline.final_equity, 10_000)
        self.assertEqual(baseline.execution_breaches, 0)
        stress = run_backtest(*self.long_inputs, slippage_rate=0.0025,
                              capacity_schedule=schedule)
        latency = run_backtest(*self.long_inputs, latency_bars=1,
                               capacity_schedule=schedule)
        self.assertLess(stress.final_equity, baseline.final_equity)
        self.assertTrue(any(item.time.hour == 8 for item in latency.trades))
        self.assertTrue(all(item.time.hour == 4 or item.time == END.to_pydatetime()
                            for item in baseline.trades))

    def test_rising_balance_is_cash_but_independent_benchmark_trades(self):
        inputs = _inputs(declining=False)
        baseline = run_backtest(*inputs)
        benchmark = run_backtest(*inputs, benchmark=True)
        self.assertEqual(baseline.final_equity, 10_000)
        self.assertFalse(baseline.trades)
        self.assertFalse(any(dict(baseline.capacity_schedule).values()))
        self.assertEqual(len(benchmark.capacity_schedule), 300)
        self.assertTrue(benchmark.trades)
        self.assertLess(benchmark.final_equity, 10_000)
        self.assertEqual(_direction(inputs[0], pd.Timestamp("2021-01-04", tz="UTC")), 0)

    def test_actual_funding_debits_long_position(self):
        observations, trade, mark, funding = self.long_inputs
        changed = {symbol: frame.copy() for symbol, frame in funding.items()}
        for frame in changed.values():
            event = frame["funding_time"] == pd.Timestamp("2021-01-05 00:00", tz="UTC")
            frame.loc[event, "rate"] = 0.001
        plain = run_backtest(*self.long_inputs)
        charged = run_backtest(observations, trade, mark, changed)
        self.assertGreater(charged.funding_usdt, 0)
        self.assertLess(charged.final_equity, plain.final_equity)


if __name__ == "__main__":
    unittest.main()
