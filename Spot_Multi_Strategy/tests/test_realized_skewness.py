import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.realized_skewness_sim import (
    SkewnessStrategyFailure, _ranking, _scores,
)
from research import realized_skewness_runner
from research.weekday_seasonality_data import WeekdayDataError, WeekdayTape


def tape_from_returns(returns):
    close = np.full((600, 8), 100.0)
    close[288:576] = 100.0 * np.exp(np.cumsum(returns, axis=0))
    close[576:] = close[575]
    return WeekdayTape(close, close, close, close,
                       np.full((600, 8), 1e6), {}, {}, {})


class RealizedSkewnessTests(unittest.TestCase):
    def test_uses_prior_day_only_and_negative_skew_is_long(self):
        returns = np.zeros((288, 8))
        returns[0] = .001
        returns[1] = -.001
        returns[2, 0] = .1
        returns[2, 7] = -.1
        tape = tape_from_returns(returns)
        first_scores = _scores(tape, 2)
        longs, shorts = _ranking(tape, 2)
        self.assertIn(7, longs)
        self.assertIn(0, shorts)
        tape.close[576:] = 1e10
        np.testing.assert_array_equal(_scores(tape, 2), first_scores)

    def test_zero_realized_variance_fails_before_trade(self):
        close = np.full((600, 8), 100.0)
        tape = WeekdayTape(close, close, close, close,
                           np.full((600, 8), 1e6), {}, {}, {})
        with self.assertRaises(SkewnessStrategyFailure):
            _scores(tape, 2)

    def test_one_shot_failure_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sealed.json"
            with patch.object(realized_skewness_runner, "RESULT", path), patch.object(
                    realized_skewness_runner, "load_tape",
                    side_effect=WeekdayDataError("source changed")):
                result = realized_skewness_runner.run_one_shot()
                self.assertEqual(result["status"], "sealed_failed_run")
                self.assertIsNone(result["metrics"])
                with self.assertRaises(FileExistsError):
                    realized_skewness_runner.run_one_shot()


if __name__ == "__main__":
    unittest.main()
