"""Synthetic checks before any formal strategy outcome is read."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research import idiosyncratic_volatility_runner
from research.idiosyncratic_volatility_sim import (
    ALT_COLS, BTC, IvolFailure, LOOKBACK_DAYS, _completed_returns, _rank,
)
from research.weekday_seasonality_data import WeekdayDataError, WeekdayTape


class IvolTests(unittest.TestCase):
    @staticmethod
    def returns():
        j = np.arange(LOOKBACK_DAYS)
        market = .025 * np.sin(.23 * j) + .012 * np.cos(.07 * j)
        returns = np.empty((LOOKBACK_DAYS, 8), dtype=float)
        returns[:, BTC] = market
        returns[:, 0] = market + .18 * np.sin(.81 * j)       # ADA high IVOL
        returns[:, 1] = .5 * market + .11 * np.cos(.59 * j)  # BCH second
        returns[:, 2] = 1.1 * market + .001 * np.sin(.43 * j)
        returns[:, 4] = .8 * market                          # ETH lowest
        returns[:, 5] = market + .02 * np.cos(.53 * j)
        returns[:, 6] = market + .04 * np.sin(.48 * j)
        returns[:, 7] = market + .03 * np.cos(.37 * j)
        return returns

    def test_residual_dispersion_sign_and_btc_exclusion(self):
        longs, shorts = _rank(self.returns(), ALT_COLS)
        self.assertEqual(set(longs), {0, 1})
        self.assertEqual(set(shorts), {2, 4})
        self.assertNotIn(BTC, longs + shorts)
        with self.assertRaises(IvolFailure):
            _rank(self.returns(), ALT_COLS + (BTC,))
        with self.assertRaises(IvolFailure):
            _rank(np.zeros((LOOKBACK_DAYS, 8)), ALT_COLS)

    def test_completed_daily_signal_does_not_read_current_or_future_bar(self):
        days = 185
        values = np.full((days * 288, 8), 100.0)
        daily_returns = self.returns()
        level = np.full(8, 100.0)
        for j in range(1, LOOKBACK_DAYS + 2):
            if j > 1:
                level = level * np.exp(daily_returns[LOOKBACK_DAYS + 1 - j])
            values[j * 288 - 1] = level
        tape = WeekdayTape(values, values.copy(), values.copy(), values.copy(),
                           np.full_like(values, 1e6), {}, {}, {})
        prior = _completed_returns(tape, 183)
        self.assertEqual(prior.shape, (LOOKBACK_DAYS, 8))
        tape.close[183 * 288:] = 1e9
        np.testing.assert_array_equal(_completed_returns(tape, 183), prior)

    def test_one_shot_source_failure_seals_and_cannot_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            with patch.object(idiosyncratic_volatility_runner, "RESULT", result_path), patch.object(
                    idiosyncratic_volatility_runner, "load_tape",
                    side_effect=WeekdayDataError("pinned source changed")):
                result = idiosyncratic_volatility_runner.run_one_shot()
                self.assertEqual(result["status"], "sealed_failed_run")
                self.assertFalse(result["gate"]["full"])
                with self.assertRaises(FileExistsError):
                    idiosyncratic_volatility_runner.run_one_shot()


if __name__ == "__main__":
    unittest.main()
