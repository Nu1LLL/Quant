"""Synthetic timing and account invariants; no historical source is opened."""

from __future__ import annotations

from datetime import date, timedelta
import unittest

import pandas as pd

from research.btc_eth_transfer_growth_source import SourceFailure
from research.us_real_yield_crypto_sim import real_yield_target, simulate


class RealYieldSyntheticTests(unittest.TestCase):
 def test_real_yield_signal_uses_only_prior_friday_and_20_observations(self):
    monday = date(2021, 1, 4)
    friday = monday - timedelta(days=3)
    valid = [friday - timedelta(days=i) for i in range(20, -1, -1)]
    yields = {day: 1.0 for day in valid}
    yields[valid[-21]] = 2.0
    yields[valid[-1]] = 1.0
    yields[monday] = -100.0
    self.assertTrue(real_yield_target(monday, yields))
    del yields[valid[-21]]
    with self.assertRaises(SourceFailure):
        real_yield_target(monday, yields)


 def test_synthetic_cash_and_basket_reconcile_without_lookahead(self):
    def frame(first: str) -> pd.DataFrame:
        idx = pd.date_range(first, "2025-12-31", tz="UTC", freq="D")
        return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                             "close": 100.0, "quote_volume": 1_000_000.0},
                            index=idx)

    prices = {"BTC": frame("2020-12-01"), "ETH": frame("2020-11-01")}
    observations = pd.date_range("2020-11-02", "2025-12-31", freq="B")
    yields = {day.date(): 2.0 - i / 10_000 for i, day in enumerate(observations)}
    run = simulate(yields, prices, {"BTC": 100.0, "ETH": 100.0})
    self.assertEqual(len(run.days), 1826)
    self.assertEqual(len([f for f in run.fills if f["side"] == "buy"]), 2)
    self.assertEqual(len([f for f in run.fills if f["side"] == "sell"]), 2)
    self.assertLess(run.terminal_cash, 10_000)
    self.assertAlmostEqual(run.days[-1].end, run.terminal_cash)
    self.assertTrue(all(f["fee"] > 0 for f in run.fills))


if __name__ == "__main__":
    unittest.main()
