"""Synthetic-only timing, currency-mark and fee checks before formal read."""

from datetime import date, timedelta
import unittest

import pandas as pd

from research.usdt_peg_sim import _target, evaluate, simulate


def synthetic_frames():
    cb_dates = pd.date_range("2021-09-01", "2026-09-30", tz="UTC", freq="D")
    bn_dates = pd.date_range("2021-08-01", "2026-09-30", tz="UTC", freq="D")
    coinbase = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                             "close": 1.0, "volume": 1000.0}, index=cb_dates)
    binance = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                            "close": 100.0, "quote_volume": 2_000_000.0},
                           index=bn_dates)
    return coinbase, binance


class UsdtPegTests(unittest.TestCase):
    def test_two_day_lag_exact_seven_day_window_and_both_tails(self):
        day = date(2021, 10, 1)
        closes = {day - timedelta(days=k): 1.0 for k in range(1, 10)}
        closes[day - timedelta(days=1)] = .90  # prohibited future window
        self.assertEqual(_target(day, closes), 1)
        closes[day - timedelta(days=8)] = 1.004
        self.assertEqual(_target(day, closes), 0)
        closes[day - timedelta(days=8)] = 1.0
        closes[day - timedelta(days=2)] = .996
        self.assertEqual(_target(day, closes), 0)
        closes[day - timedelta(days=2)] = .997
        self.assertEqual(_target(day, closes), 1)

    def test_flat_btc_account_pays_both_fills_and_reconciles(self):
        coinbase, binance = synthetic_frames()
        primary = simulate(coinbase, binance, 100.0)
        gross = simulate(coinbase, binance, 100.0, slippage=0, fee_rate=0)
        self.assertEqual(len(primary.usd_days), 1826)
        self.assertEqual(len(primary.fills), 2)
        self.assertEqual(primary.fills[0]["side"], "buy")
        self.assertEqual(primary.fills[-1]["side"], "sell")
        self.assertLess(primary.terminal_cash_usdt, 10_000)
        self.assertAlmostEqual(gross.terminal_cash_usdt, 10_000, places=6)
        self.assertAlmostEqual(primary.usd_days[-1].end,
                               primary.terminal_cash_usdt, places=7)

    def test_usdt_cash_is_marked_to_usd_during_depeg(self):
        coinbase, binance = synthetic_frames()
        stamp = pd.Timestamp("2021-10-01", tz="UTC")
        coinbase.loc[stamp, "close"] = .99
        coinbase.loc[stamp, "low"] = .99
        cash = simulate(coinbase, binance, 100.0, mode="cash")
        self.assertEqual(cash.usdt_days[0].end, 10_000)
        self.assertEqual(cash.usd_days[0].end, 9_900)
        self.assertEqual(cash.fills, ())

    def test_full_synthetic_gate_runs_and_never_claims_oos(self):
        coinbase, binance = synthetic_frames()
        base = simulate(coinbase, binance, 100.0)
        gross = simulate(coinbase, binance, 100.0, slippage=0, fee_rate=0)
        double = simulate(coinbase, binance, 100.0, slippage=.001)
        delayed = simulate(coinbase, binance, 100.0, latency_days=1)
        hold = simulate(coinbase, binance, 100.0, mode="buy_hold")
        cash = simulate(coinbase, binance, 100.0, mode="cash")
        result = evaluate(base, gross, double, delayed, hold, cash, binance)
        self.assertEqual(result["usd_full"]["days"], 1826)
        self.assertEqual(result["usd_monte_carlo"]["paths"], 5000)
        self.assertFalse(result["checks"]["independent_executable_oos"])
        self.assertFalse(result["all_gate_passed"])


if __name__ == "__main__":
    unittest.main()
