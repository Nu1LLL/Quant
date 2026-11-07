"""Synthetic old-holder rule, 4h execution, and full JSON report tests."""

from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
import json
import unittest

import numpy as np
import pandas as pd

from research.old_coin_spend_sim import END, _direction, run_backtest
from research.old_coin_spend_source import FIRST, LAST, OldCoinObservation
from research.old_coin_spend_metrics import evaluate
from research.run_old_coin_spend_crypto import _serialize_payload
from tests.test_tips_real_yield_sim import _fixture as market_fixture


def _inputs(*, declining: bool):
    _, old_trade, old_mark, old_funding = market_fixture()
    trade, mark, funding = {}, {}, {}
    for symbol in old_trade:
        trade[symbol] = old_trade[symbol].loc[:END].copy()
        mark[symbol] = old_mark[symbol].loc[:END].copy()
        for name in ("high", "low", "close", "quote_volume"):
            trade[symbol].loc[END, name] = np.nan
        for name in ("high", "low", "close"):
            mark[symbol].loc[END, name] = np.nan
        funding[symbol] = old_funding[symbol].loc[
            old_funding[symbol]["funding_time"] <= END].reset_index(drop=True)
    observations = []
    day = FIRST
    i = 0
    while day <= LAST:
        value = Decimal(100_000 - i if declining else 100_000 + i)
        observations.append(OldCoinObservation(
            day, value, datetime.combine(day + timedelta(days=4), time.min,
                                         timezone.utc)))
        day += timedelta(days=1)
        i += 1
    return tuple(observations), trade, mark, funding


class OldCoinSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = _inputs(declining=True)

    def test_decreasing_old_coin_spend_buys_with_replayed_stress(self):
        baseline = run_backtest(*self.inputs)
        schedule = dict(baseline.capacity_schedule)
        self.assertEqual(len(schedule), 287)
        self.assertTrue(all(schedule.values()))
        self.assertEqual(len(baseline.records), 2007)
        self.assertTrue(baseline.terminal_flat)
        self.assertEqual(baseline.execution_breaches, 0)
        self.assertLess(baseline.final_equity, 10_000)
        stress = run_backtest(*self.inputs, slippage_rate=0.0025,
                              capacity_schedule=schedule)
        latency = run_backtest(*self.inputs, latency_bars=1,
                               capacity_schedule=schedule)
        self.assertLess(stress.final_equity, baseline.final_equity)
        self.assertTrue(any(item.time.hour == 8 for item in latency.trades))
        self.assertEqual(_direction(self.inputs[0], pd.Timestamp("2021-01-04", tz="UTC")), 1)

    def test_increasing_old_coin_spend_cash_and_independent_benchmark(self):
        inputs = _inputs(declining=False)
        baseline = run_backtest(*inputs)
        benchmark = run_backtest(*inputs, benchmark=True)
        self.assertEqual(baseline.final_equity, 10_000)
        self.assertFalse(baseline.trades)
        self.assertTrue(benchmark.trades)
        self.assertEqual(_direction(inputs[0], pd.Timestamp("2021-01-04", tz="UTC")), 0)

    def test_funding_and_complete_report_serialization(self):
        observations, trade, mark, funding = self.inputs
        changed = {symbol: frame.copy() for symbol, frame in funding.items()}
        for frame in changed.values():
            event = frame["funding_time"] == pd.Timestamp("2021-01-05 00:00", tz="UTC")
            frame.loc[event, "rate"] = 0.001
        baseline = run_backtest(*self.inputs)
        charged = run_backtest(observations, trade, mark, changed)
        self.assertGreater(charged.funding_usdt, 0)
        self.assertLess(charged.final_equity, baseline.final_equity)
        benchmark = run_backtest(*self.inputs, benchmark=True)
        report = evaluate(baseline, benchmark, trade["BTCUSDT"], None, None)
        payload = {"status": "synthetic", "passed": np.bool_(False),
                   "score": np.float64(0.1), "diagnostics": report}
        recovered = json.loads(_serialize_payload(payload))
        self.assertEqual(len(recovered["diagnostics"]["annual_walk_forward"]), 6)
        self.assertEqual(len(recovered["diagnostics"]["monthly"]["returns"]), 66)
        self.assertIs(recovered["passed"], False)


if __name__ == "__main__":
    unittest.main()
