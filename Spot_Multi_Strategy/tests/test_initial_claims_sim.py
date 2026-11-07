"""Synthetic weekly claims and market ledgers; no formal outcome is read."""

from datetime import datetime, time, timedelta, timezone
import json
import unittest

import numpy as np
import pandas as pd

from research.initial_claims_sim import END, _direction, run_backtest
from research.initial_claims_source import ClaimsObservation, FIRST, ROWS
from research.run_initial_claims_crypto import _serialize_payload
from research.tips_real_yield_metrics import evaluate
from tests.test_tips_real_yield_sim import _fixture as market_fixture


def _inputs(*, improving: bool):
    _, trade, mark, funding = market_fixture()
    observations = []
    for i in range(ROWS):
        stamp = FIRST + timedelta(days=7*i)
        count = 300_000 - i*100 if improving else 300_000 + i*100
        observations.append(ClaimsObservation(
            stamp, count,
            datetime.combine(stamp + timedelta(days=9), time.min, timezone.utc)))
    return tuple(observations), trade, mark, funding


class ClaimsSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = _inputs(improving=True)

    def test_improving_claims_buys_and_stress_reuses_schedule(self):
        baseline = run_backtest(*self.inputs)
        schedule = dict(baseline.capacity_schedule)
        self.assertEqual(len(schedule), 300)
        self.assertTrue(all(schedule.values()))
        self.assertEqual(len(baseline.records), 2099)
        self.assertTrue(baseline.terminal_flat)
        self.assertEqual(baseline.execution_breaches, 0)
        self.assertLess(baseline.final_equity, 10_000)
        stress = run_backtest(*self.inputs, slippage_rate=0.0025,
                              capacity_schedule=schedule)
        latency = run_backtest(*self.inputs, latency_bars=1,
                               capacity_schedule=schedule)
        self.assertLess(stress.final_equity, baseline.final_equity)
        self.assertTrue(any(item.time.hour == 8 for item in latency.trades))
        self.assertTrue(all(item.time.hour == 4 or item.time == END.to_pydatetime()
                            for item in baseline.trades))
        self.assertEqual(_direction(self.inputs[0], pd.Timestamp("2021-01-04", tz="UTC")), 1)

    def test_worsening_claims_cash_and_independent_benchmark(self):
        inputs = _inputs(improving=False)
        baseline = run_backtest(*inputs)
        benchmark = run_backtest(*inputs, benchmark=True)
        self.assertEqual(baseline.final_equity, 10_000)
        self.assertFalse(baseline.trades)
        self.assertTrue(benchmark.trades)
        self.assertEqual(_direction(inputs[0], pd.Timestamp("2021-01-04", tz="UTC")), 0)

    def test_funding_and_complete_report_json_roundtrip(self):
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
        self.assertEqual(set(recovered["diagnostics"]["checks"]),
                         set(report["checks"]))
        self.assertIs(recovered["passed"], False)
        self.assertEqual(len(recovered["diagnostics"]["annual_walk_forward"]), 6)
        self.assertEqual(len(recovered["diagnostics"]["monthly"]["returns"]), 69)


if __name__ == "__main__":
    unittest.main()
