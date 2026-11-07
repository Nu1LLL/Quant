import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.btc_eth_pairs_metrics import _metrics_from_factors
from research.weekly_illiquidity_sim import (
    IlliquidityFailure, _WeeklyAccount, _rank, _score,
)
from research import weekly_illiquidity_runner
from research.weekday_seasonality_data import WeekdayDataError, WeekdayTape


def synthetic_tape(days=40):
    price = np.full((days * 288, 8), 100.0)
    return WeekdayTape(price, price.copy(), price.copy(), price.copy(),
                       np.full_like(price, 1e6), {}, {}, {})


class WeeklyIlliquidityTests(unittest.TestCase):
    def test_score_uses_only_28_completed_days_and_high_illiquidity_is_long(self):
        tape = synthetic_tape()
        tape.close[10 * 288 - 1, 0] = 101.0
        tape.close[12 * 288 - 1, 0] = 99.0
        before = _score(tape, 30)
        self.assertGreater(before[0], before[1])
        self.assertIn(0, _rank(tape, 30)[0])
        tape.close[30 * 288:, :] = 1e9
        np.testing.assert_array_equal(_score(tape, 30), before)

    def test_zero_daily_quote_volume_fails(self):
        tape = synthetic_tape()
        tape.quote_volume[8 * 288:9 * 288, 1] = 0.0
        with self.assertRaises(IlliquidityFailure):
            _score(tape, 30)

    def test_weekly_account_reconciles_fills_fee_and_funding_without_daily_churn(self):
        tape = synthetic_tape()
        account = _WeeklyAccount("base_1x", .0005, 1, False)
        account.trade_week(tape, 30, (0, 1), (2, 3))
        self.assertEqual(len(account.records), 7)
        self.assertEqual(len(account.trades), 1)
        trade = account.trades[0]
        self.assertAlmostEqual(account.equity - 10_000,
                               trade.executed_price_pnl + trade.fee_pnl + trade.funding_pnl,
                               places=8)
        self.assertAlmostEqual(account.records[0].equity_start, 10_000)
        self.assertAlmostEqual(account.records[-1].equity_end, account.equity)
        self.assertLess(account.equity, 10_000)
        self.assertEqual(sum(row.equity_end != row.equity_start for row in account.records), 2)

    def test_funding_event_is_charged_once_inside_the_held_week(self):
        tape = synthetic_tape()
        event = 30 * 288 + 12
        tape.funding[event] = ((0, .001),)
        account = _WeeklyAccount("base_1x", .0005, 1, False)
        account.trade_week(tape, 30, (0, 1), (2, 3))
        self.assertLess(account.trades[0].funding_pnl, 0)
        self.assertAlmostEqual(account.equity - 10_000,
                               account.trades[0].executed_price_pnl
                               + account.trades[0].fee_pnl
                               + account.trades[0].funding_pnl, places=8)

    def test_compressed_daily_path_reproduces_five_minute_drawdown(self):
        tape = synthetic_tape()
        start = 30 * 288
        tape.close[start + 25:start + 50, 0] = 90.0
        tape.low[start + 25:start + 50, 0] = 85.0
        account = _WeeklyAccount("base_1x", .0005, 1, False)
        account.trade_week(tape, 30, (0, 1), (2, 3))
        factors = np.array([
            (row.equity_end / row.equity_start, row.max_regular_factor,
             row.min_adverse_factor, row.max_intraday_drawdown)
            for row in account.records
        ])
        replay = _metrics_from_factors(factors)
        self.assertAlmostEqual(replay["ending_equity"], account.equity, places=8)
        self.assertAlmostEqual(replay["max_drawdown"], account.max_drawdown, places=8)

    def test_one_shot_failure_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sealed.json"
            with patch.object(weekly_illiquidity_runner, "RESULT", path), patch.object(
                    weekly_illiquidity_runner, "load_tape",
                    side_effect=WeekdayDataError("source changed")):
                outcome = weekly_illiquidity_runner.run_one_shot()
                self.assertEqual(outcome["status"], "sealed_failed_run")
                self.assertIsNone(outcome["metrics"])
                with self.assertRaises(FileExistsError):
                    weekly_illiquidity_runner.run_one_shot()


if __name__ == "__main__":
    unittest.main()
