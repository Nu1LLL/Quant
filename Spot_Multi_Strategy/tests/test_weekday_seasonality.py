import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.weekday_seasonality_data import (
    BASE_MS, WeekdayDataError, WeekdayTape, _funding_bar_index,
)
from research.weekday_seasonality_sim import _Account, _ranking
from research import weekday_seasonality_runner


class WeekdaySeasonalityTests(unittest.TestCase):
    def test_subsecond_funding_timestamp_maps_to_utc_hour(self):
        self.assertEqual(_funding_bar_index(BASE_MS + 2, 288), 0)
        self.assertEqual(_funding_bar_index(BASE_MS + 8 * 3_600_000 + 6, 288), 96)
        with self.assertRaises(WeekdayDataError):
            _funding_bar_index(BASE_MS + 1_000, 288)

    def test_rank_uses_only_twenty_prior_same_weekdays(self):
        returns = np.zeros((150, 8), dtype=float)
        target = 140
        for week in range(1, 21):
            returns[target - 7 * week] = np.arange(8) / 100
        returns[target - 1] = np.arange(8)[::-1] * 100.0
        longs, shorts = _ranking(returns, target)
        self.assertEqual(longs, (6, 7))
        self.assertEqual(shorts, (0, 1))

    def test_funding_credit_suppression_and_cost_reconciliation(self):
        opening = np.full((600, 8), 100.0)
        tape = WeekdayTape(opening, opening * 1.01, opening * .99,
                           opening.copy(), np.full((600, 8), 1e6),
                           {96: ((2, .001),)}, {}, {})
        base = _Account("base", .0005, 1, False)
        stress = _Account("stress", .0005, 1, True)
        for account in (base, stress):
            account.execute(tape, 0, (0, 1), (2, 3), "bear_low")
            trade = account.trades[0]
            self.assertAlmostEqual(trade.net_pnl,
                                   trade.gross_price_pnl + trade.fee_pnl + trade.funding_pnl)
            self.assertEqual(len(account.records), 1)
        self.assertGreater(base.equity, stress.equity)
        self.assertGreater(base.trades[0].funding_pnl, 0)
        self.assertEqual(stress.trades[0].funding_pnl, 0)

    def test_failed_run_is_sealed_once(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            with patch.object(weekday_seasonality_runner, "RESULT", path), patch.object(
                    weekday_seasonality_runner, "load_tape",
                    side_effect=WeekdayDataError("bad pinned source")):
                result = weekday_seasonality_runner.run_one_shot()
                self.assertEqual(result["status"], "sealed_failed_run")
                self.assertIsNone(result["metrics"])
                with self.assertRaises(FileExistsError):
                    weekday_seasonality_runner.run_one_shot()


if __name__ == "__main__":
    unittest.main()
