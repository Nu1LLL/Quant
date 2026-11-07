"""Synthetic weekend timing, long/short risk, funding, and report serialization."""

import json
import unittest

import numpy as np
import pandas as pd

from research.weekend_reentry_sim import (
    END, SYMBOLS, WARMUP, MarginRuinError, _weekend_weights, run_backtest,
)
from research.weekend_reentry_metrics import evaluate
from research.run_weekend_reentry_crypto import _serialize_payload
from tests.test_tips_real_yield_sim import _fixture as market_fixture


def _inputs():
    _, trade, mark, funding = market_fixture()
    return trade, mark, funding


class WeekendReentryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = _inputs()

    def test_weekend_ranking_uses_completed_bars_and_ties(self):
        grid = pd.date_range(WARMUP, END, freq="4h", tz="UTC")
        monday = pd.Timestamp("2021-01-04 00:00", tz="UTC")
        index = int(grid.get_loc(monday))
        arrays = {symbol: {"open": np.ones(len(grid)),
                           "close": np.ones(len(grid))} for symbol in SYMBOLS}
        desired = {"BTCUSDT": -0.10, "ETHUSDT": 0.10, "DOGEUSDT": 0.0,
                   "LTCUSDT": 0.20, "BCHUSDT": -0.20}
        for symbol, ret in desired.items():
            arrays[symbol]["close"][index - 1] = 1 + ret
        weights = _weekend_weights(arrays, grid, monday)
        self.assertEqual(weights, {"BTCUSDT": 0.25, "ETHUSDT": -0.25,
                                   "DOGEUSDT": 0.0, "LTCUSDT": -0.25,
                                   "BCHUSDT": 0.25})
        arrays["DOGEUSDT"]["close"][index] = 100.0
        self.assertEqual(_weekend_weights(arrays, grid, monday), weights)
        for symbol in SYMBOLS:
            arrays[symbol]["close"][index - 1] = 1.0
        self.assertEqual(_weekend_weights(arrays, grid, monday),
                         {"BTCUSDT": 0.25, "ETHUSDT": -0.25,
                          "DOGEUSDT": 0.0, "LTCUSDT": -0.25,
                          "BCHUSDT": 0.25})

    def test_one_day_holding_funding_and_replayed_stresses(self):
        baseline = run_backtest(*self.inputs)
        schedule = dict(baseline.capacity_schedule)
        self.assertEqual(len(schedule), 300)
        self.assertTrue(all(schedule.values()))
        self.assertTrue(baseline.terminal_flat)
        self.assertEqual(baseline.execution_breaches, 0)
        self.assertLess(baseline.final_equity, 10_000)
        self.assertTrue(any(item.side == "buy" for item in baseline.trades))
        self.assertTrue(any(item.side == "sell" for item in baseline.trades))
        self.assertTrue(all(item.time.weekday() in (0, 1) and item.time.hour == 4
                            for item in baseline.trades))
        stress = run_backtest(*self.inputs, slippage_rate=0.0025,
                              capacity_schedule=schedule)
        delayed = run_backtest(*self.inputs, latency_bars=1,
                               capacity_schedule=schedule)
        self.assertLess(stress.final_equity, baseline.final_equity)
        self.assertTrue(all(item.time.hour == 8 for item in delayed.trades))
        trade, mark, funding = self.inputs
        changed = {symbol: frame.copy() for symbol, frame in funding.items()}
        event = changed["BTCUSDT"]["funding_time"] == pd.Timestamp(
            "2021-01-05 00:00", tz="UTC")
        changed["BTCUSDT"].loc[event, "rate"] = 0.001
        charged = run_backtest(trade, mark, changed)
        self.assertGreater(charged.funding_usdt, baseline.funding_usdt)
        self.assertLess(charged.final_equity, baseline.final_equity)

    def test_short_adverse_mark_breach_and_complete_json(self):
        trade, mark, funding = self.inputs
        changed = {symbol: frame.copy() for symbol, frame in mark.items()}
        changed["LTCUSDT"].loc[pd.Timestamp("2021-01-04 04:00", tz="UTC"), "high"] = 1000.0
        with self.assertRaises(MarginRuinError):
            run_backtest(trade, changed, funding)
        baseline = run_backtest(*self.inputs)
        benchmark = run_backtest(*self.inputs, benchmark=True)
        report = evaluate(baseline, benchmark, trade["BTCUSDT"], None, None)
        payload = {"numpy_boolean": np.bool_(False),
                   "numpy_float": np.float64(0.1), "diagnostics": report}
        recovered = json.loads(_serialize_payload(payload))
        self.assertEqual(len(recovered["diagnostics"]["annual_walk_forward"]), 6)
        self.assertEqual(len(recovered["diagnostics"]["monthly"]["returns"]), 69)
        self.assertIs(recovered["numpy_boolean"], False)

    def test_capacity_blocks_whole_entry_but_does_not_cancel_exit(self):
        trade, mark, funding = self.inputs
        blocked_entry = {symbol: frame.copy() for symbol, frame in trade.items()}
        sunday = (blocked_entry["LTCUSDT"].index >= pd.Timestamp(
            "2021-01-03 00:00", tz="UTC")) & (blocked_entry["LTCUSDT"].index <
                                             pd.Timestamp("2021-01-04 00:00", tz="UTC"))
        blocked_entry["LTCUSDT"].loc[sunday, "quote_volume"] = 0.0
        skipped = run_backtest(blocked_entry, mark, funding)
        self.assertFalse(dict(skipped.capacity_schedule)[pd.Timestamp(
            "2021-01-04", tz="UTC").date()])
        self.assertFalse(any(item.time == pd.Timestamp("2021-01-04 04:00", tz="UTC")
                             for item in skipped.trades))
        self.assertGreater(skipped.capacity_breaches, 0)

        blocked_exit = {symbol: frame.copy() for symbol, frame in trade.items()}
        monday = (blocked_exit["LTCUSDT"].index >= pd.Timestamp(
            "2021-01-04 00:00", tz="UTC")) & (blocked_exit["LTCUSDT"].index <
                                             pd.Timestamp("2021-01-05 00:00", tz="UTC"))
        blocked_exit["LTCUSDT"].loc[monday, "quote_volume"] = 0.0
        exiting = run_backtest(blocked_exit, mark, funding)
        self.assertTrue(any(item.time == pd.Timestamp("2021-01-05 04:00", tz="UTC")
                            and item.symbol == "LTCUSDT" for item in exiting.trades))
        self.assertGreater(exiting.capacity_breaches, 0)


if __name__ == "__main__":
    unittest.main()
