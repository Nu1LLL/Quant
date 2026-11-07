import math
import unittest

import numpy as np

from research.btc_quarter_hour_open_sim import _folds, day_profile, path_metrics


class QuarterHourReplayTests(unittest.TestCase):
    def test_flat_day_pays_cost_72_times(self) -> None:
        opening = np.full(288, 100.0)
        low = opening.copy()
        cost = (1 - .0001) * (1 - .0005) / ((1 + .0001) * (1 + .0005))
        for lag in (0, 1):
            factor, worst, best, drawdown, gross = day_profile(opening, low, lag_bars=lag)
            self.assertAlmostEqual(factor, cost ** 72)
            self.assertAlmostEqual(gross, 1.0)
            self.assertAlmostEqual(worst, factor)
            self.assertEqual(best, 1.0)
            self.assertAlmostEqual(drawdown, 1 - factor)

    def test_day_mark_uses_held_bar_low(self) -> None:
        opening = np.full(288, 100.0)
        low = opening.copy()
        low[3] = 80.0
        factor, worst, _, dd, _ = day_profile(opening, low, fee=0, slippage=0)
        self.assertEqual(factor, 1.0)
        self.assertEqual(worst, .8)
        self.assertAlmostEqual(dd, .2)

    def test_metric_drawdown_includes_intraday_low(self) -> None:
        factors = np.array([1.0, 1.0])
        minima = np.array([.8, 1.0])
        maxima = np.ones(2)
        local = np.array([.2, 0.0])
        m = path_metrics(factors, minima, maxima, local)
        self.assertAlmostEqual(m["max_drawdown"], .2)
        self.assertEqual(m["final_equity"], 10_000.0)
        self.assertTrue(math.isfinite(m["cagr"]))

    def test_calendar_folds_use_exact_year_boundaries(self) -> None:
        factor = np.full(2099, 1.00001)
        zeros = np.zeros(2099)
        folds = _folds(factor, factor, factor, zeros)
        self.assertAlmostEqual(folds["2021"]["final_equity"], 10_000 * 1.00001 ** 365)
        self.assertAlmostEqual(folds["2024"]["final_equity"], 10_000 * 1.00001 ** 366)
        self.assertAlmostEqual(folds["2026"]["final_equity"], 10_000 * 1.00001 ** 273)


if __name__ == "__main__":
    unittest.main()
