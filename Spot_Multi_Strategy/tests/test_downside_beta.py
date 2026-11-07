import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research import downside_beta_runner
from research.downside_beta_sim import _rank, _weekly_returns
from research.weekday_seasonality_data import WeekdayDataError, WeekdayTape


def conditional_tape():
    n = 191 * 288
    prices = np.full((n, 8), 100.0)
    market = np.array([-(.025 + .002 * (j % 5)) if j % 2 else
                       (.02 + .003 * (j % 4)) for j in range(26)])
    returns = np.zeros((26, 8))
    returns[:, 3] = market
    returns[:, 0] = np.where(market < 0, 2.0 * market, .5 * market)
    returns[:, 1] = .25 * market
    returns[:, 2] = .5 * market
    returns[:, 4] = .75 * market
    returns[:, 5] = 1.1 * market
    returns[:, 6] = 1.2 * market
    returns[:, 7] = 1.3 * market
    level = np.full(8, 100.0)
    for j in range(27):
        day = 7 + j * 7
        prices[day * 288 - 1] = level
        if j < 26:
            level = level * np.exp(returns[j])
    return WeekdayTape(prices, prices.copy(), prices.copy(), prices.copy(),
                       np.full_like(prices, 1e6), {}, {}, {})


class DownsideBetaTests(unittest.TestCase):
    def test_completed_weeks_and_downside_sign(self):
        tape = conditional_tape()
        weekly = _weekly_returns(tape, 189)
        self.assertEqual(weekly.shape, (26, 8))
        self.assertIn(0, _rank(tape, 189)[0])
        before = weekly.copy()
        tape.close[189 * 288:] = 1e9
        np.testing.assert_array_equal(_weekly_returns(tape, 189), before)

    def test_fewer_than_five_negative_weeks_is_cash(self):
        tape = conditional_tape()
        for day in range(7, 190, 7):
            tape.close[day * 288 - 1, 3] = 100.0 * (1.01 ** (day // 7))
        self.assertIsNone(_rank(tape, 189))

    def test_one_shot_failure_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sealed.json"
            with patch.object(downside_beta_runner, "RESULT", path), patch.object(
                    downside_beta_runner, "load_tape",
                    side_effect=WeekdayDataError("source changed")):
                result = downside_beta_runner.run_one_shot()
                self.assertEqual(result["status"], "sealed_failed_run")
                with self.assertRaises(FileExistsError):
                    downside_beta_runner.run_one_shot()


if __name__ == "__main__":
    unittest.main()
