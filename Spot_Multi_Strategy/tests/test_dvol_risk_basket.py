"""Synthetic timing and accounting checks; no historical strategy outcome."""

import math
import unittest

import numpy as np
import pandas as pd

from research.dvol_risk_sim import simulate, weekly_targets


def fixture(iv: float = 3.0):
    dates = pd.date_range("2021-03-01", "2026-09-30", tz="UTC", freq="D")
    vol_dates = pd.date_range("2021-03-24", "2026-09-30", tz="UTC", freq="D")
    idx = np.arange(len(dates))
    close = 100 + 15 * np.sin(idx * .3) + idx * .01
    prices = {}
    dvol = {}
    for c, multiplier in (("BTC", 1.0), ("ETH", 2.0)):
        values = close * multiplier
        prices[c + "USDT"] = pd.DataFrame({"open": values, "high": values * 1.01,
                                            "low": values * .99, "close": values,
                                            "quote_volume": 100_000_000.0}, index=dates)
        dvol[c] = pd.DataFrame({"open": iv, "high": iv, "low": iv,
                                "close": iv}, index=vol_dates)
    terminal = {s: float(frame.iloc[-1]["close"]) for s, frame in prices.items()}
    return dvol, prices, terminal


class DvolRiskBasketTest(unittest.TestCase):
    def test_first_decision_uses_only_prior_completed_days(self):
        dvol, prices, _ = fixture()
        initial = weekly_targets(dvol, prices)
        first = next(iter(initial))
        self.assertEqual(str(first), "2021-05-05")
        dvol["BTC"].loc[pd.Timestamp("2021-05-03", tz="UTC"), "close"] = 200.0
        prices["BTCUSDT"].loc[pd.Timestamp("2021-05-03", tz="UTC"), "close"] *= 2
        changed = weekly_targets(dvol, prices)
        self.assertEqual(initial[first], changed[first])
        self.assertEqual(len(initial), 283)

    def test_buy_is_wednesday_and_terminal_cash_reconciles(self):
        dvol, prices, terminal = fixture()
        result = simulate(dvol, prices, terminal)
        self.assertEqual(len(result.days), 1979)
        self.assertEqual(result.entries, 1)
        self.assertEqual(result.trades[0]["day"], "2021-05-05")
        self.assertEqual(result.trades[-1]["day"], "2026-10-01")
        self.assertTrue(result.trades[-1]["forced"])
        self.assertTrue(math.isclose(result.final, result.days[-1].end))
        self.assertGreater(result.fees, 0)

    def test_expensive_options_keep_portfolio_in_cash(self):
        dvol, prices, terminal = fixture(iv=250.0)
        result = simulate(dvol, prices, terminal)
        self.assertEqual(result.entries, 0)
        self.assertEqual(result.trades, ())
        self.assertEqual(result.final, 10_000.0)


if __name__ == "__main__":
    unittest.main()
