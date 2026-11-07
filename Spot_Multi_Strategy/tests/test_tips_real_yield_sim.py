"""Synthetic 4h ledger tests; no Treasury or Binance outcome is read."""

from datetime import date, datetime, timedelta, timezone
import unittest

import numpy as np
import pandas as pd

from research.tips_real_yield_sim import END, START, SYMBOLS, WARMUP, run_backtest
from research.tips_real_yield_source import YieldObservation


PRICES = {"BTCUSDT": 50_000.0, "ETHUSDT": 2_000.0, "DOGEUSDT": 0.1,
          "LTCUSDT": 100.0, "BCHUSDT": 100.0}


def _fixture():
    grid = pd.date_range(WARMUP, END, freq="4h", tz="UTC")
    funding_times = pd.date_range(WARMUP, END, freq="8h", tz="UTC")
    trade, mark, funding = {}, {}, {}
    for symbol in SYMBOLS:
        price = PRICES[symbol]
        data = {name: np.full(len(grid), price) for name in ("open", "high", "low", "close")}
        data["quote_volume"] = np.full(len(grid), 1_000_000_000.0)
        for name in ("high", "low", "close", "quote_volume"):
            data[name][-1] = np.nan
        trade[symbol] = pd.DataFrame(data, index=grid)
        mark[symbol] = trade[symbol].drop(columns="quote_volume").copy()
        funding[symbol] = pd.DataFrame({
            "funding_time": funding_times,
            "raw_funding_time": funding_times + pd.Timedelta(milliseconds=1),
            "rate": np.zeros(len(funding_times)),
            "funding_interval_hours": np.full(len(funding_times), 8.0),
        })
    observations = []
    day = date(2020, 11, 1)
    sequence = 0
    while day <= date(2026, 9, 30):
        if day.weekday() < 5:
            observations.append(YieldObservation(
                day, 1.0 - sequence * 0.0001,
                datetime.combine(day + timedelta(days=2), datetime.min.time(), timezone.utc)))
            sequence += 1
        day += timedelta(days=1)
    return tuple(observations), trade, mark, funding


class TipsRealYieldSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = _fixture()

    def test_fixed_weekly_ledger_and_replayed_stresses(self):
        baseline = run_backtest(*self.inputs)
        self.assertEqual(len(baseline.records), 2099)
        self.assertEqual(baseline.records[0].day, START.date())
        self.assertEqual(baseline.records[-1].day, date(2026, 9, 30))
        self.assertTrue(baseline.terminal_flat)
        self.assertEqual(baseline.capacity_breaches, 0)
        self.assertEqual(baseline.execution_breaches, 0)
        self.assertLess(baseline.final_equity, 10_000.0)  # flat price, positive costs
        self.assertGreater(baseline.final_equity, 9_000.0)
        self.assertTrue(all(trade.time.hour == 4 or trade.time == END.to_pydatetime()
                            for trade in baseline.trades))
        schedule = dict(baseline.capacity_schedule)
        self.assertTrue(schedule)
        stress = run_backtest(*self.inputs, slippage_rate=0.0025,
                              capacity_schedule=schedule)
        delayed = run_backtest(*self.inputs, latency_bars=1,
                               capacity_schedule=schedule)
        benchmark = run_backtest(*self.inputs, benchmark=True,
                                 capacity_schedule=schedule)
        self.assertLess(stress.final_equity, baseline.final_equity)
        self.assertTrue(any(trade.time.hour == 8 for trade in delayed.trades))
        self.assertEqual(benchmark.final_equity, baseline.final_equity)

    def test_actual_funding_debits_long_position(self):
        observations, trade, mark, funding = self.inputs
        changed = {symbol: frame.copy() for symbol, frame in funding.items()}
        for frame in changed.values():
            event = frame["funding_time"] == pd.Timestamp("2021-01-05 00:00", tz="UTC")
            frame.loc[event, "rate"] = 0.001
        without = run_backtest(*self.inputs)
        with_funding = run_backtest(observations, trade, mark, changed)
        self.assertGreater(with_funding.funding_usdt, 0)
        self.assertLess(with_funding.final_equity, without.final_equity)

    def test_zero_change_weeks_keep_benchmark_on_same_cash_schedule(self):
        observations, trade, mark, funding = self.inputs
        constant = tuple(YieldObservation(row.observed_date, 1.0, row.proxy_available_at)
                         for row in observations)
        baseline = run_backtest(constant, trade, mark, funding)
        benchmark = run_backtest(constant, trade, mark, funding,
                                 benchmark=True,
                                 capacity_schedule=dict(baseline.capacity_schedule))
        self.assertFalse(any(dict(baseline.capacity_schedule).values()))
        self.assertEqual(baseline.final_equity, 10_000.0)
        self.assertEqual(benchmark.final_equity, 10_000.0)
        self.assertFalse(benchmark.trades)


if __name__ == "__main__":
    unittest.main()
