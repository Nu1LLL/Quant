"""Synthetic-only tests: no registered market outcome is read here."""

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))

from research.altcoin_reversal_backtest import (
    DataIntegrityError,
    FIXED_SYMBOLS,
    Signal,
    _audit_bars,
    _metrics,
    _monte_carlo,
    _select_for_day,
    _simulate,
    run_reversal_backtest,
)


def _constant_bars(start: str, end: str, price: float = 100.0) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="4h", inclusive="left", tz="UTC")
    return pd.DataFrame({
        "open": price, "high": price, "low": price, "close": price,
        "quote_volume": 2_000_000.0,
    }, index=index)


class ReversalBacktestTests(unittest.TestCase):
    def test_signal_uses_only_yesterday_and_prior_30_day_volume(self) -> None:
        dates = pd.date_range("2020-10-01", "2021-01-11", freq="D", tz="UTC")
        day = pd.Timestamp("2021-01-10", tz="UTC")
        closes = {
            "AAVEUSDT": 90.0, "BANDUSDT": 95.0, "COMPUSDT": 99.0,
            "DASHUSDT": 100.0, "IOTAUSDT": 101.0, "KSMUSDT": 105.0,
            "ONTUSDT": 110.0,
        }
        daily = {}
        for symbol in FIXED_SYMBOLS:
            frame = pd.DataFrame({"volume": 10_000_000.0, "close": 100.0}, index=dates)
            if symbol in closes:
                frame.loc[day - pd.Timedelta(days=1), "volume"] = 5_500_000.0
                frame.loc[day - pd.Timedelta(days=1), "close"] = closes[symbol]
            daily[symbol] = frame
        signal = _select_for_day(day, daily, FIXED_SYMBOLS)
        self.assertEqual(signal.eligible_count, 14)
        self.assertEqual(signal.low_activity_count, 7)
        self.assertEqual(signal.longs, ("AAVEUSDT", "BANDUSDT"))
        self.assertEqual(signal.shorts, ("ONTUSDT", "KSMUSDT"))
        self.assertEqual(signal.reason, "trade")
        # The current day's price/volume cannot influence a decision made at 00.
        for frame in daily.values():
            frame.loc[day, "close"] = 100_000.0
            frame.loc[day, "volume"] = 1.0
        self.assertEqual(_select_for_day(day, daily, FIXED_SYMBOLS), signal)

    def test_equal_return_ties_are_alphabetic_and_disjoint(self) -> None:
        dates = pd.date_range("2020-10-01", "2021-01-11", freq="D", tz="UTC")
        day = pd.Timestamp("2021-01-10", tz="UTC")
        daily = {}
        for symbol in FIXED_SYMBOLS:
            frame = pd.DataFrame({"volume": 10_000_000.0, "close": 100.0}, index=dates)
            if symbol in FIXED_SYMBOLS[:8]:
                frame.loc[day - pd.Timedelta(days=1), "volume"] = 5_500_000.0
            daily[symbol] = frame
        signal = _select_for_day(day, daily, FIXED_SYMBOLS)
        self.assertEqual(signal.longs, ("AAVEUSDT", "BANDUSDT"))
        self.assertEqual(signal.shorts, ("COMPUSDT", "DASHUSDT"))

    def test_04_funding_milliseconds_settle_before_rebalance(self) -> None:
        start = pd.Timestamp("2021-01-01", tz="UTC")
        end = pd.Timestamp("2021-01-03", tz="UTC")
        symbols = ("AAVEUSDT", "BANDUSDT", "COMPUSDT", "DASHUSDT")
        trade = {symbol: _constant_bars(str(start), str(end)) for symbol in symbols}
        mark = {symbol: trade[symbol].drop(columns="quote_volume") for symbol in symbols}
        funding = {symbol: pd.DataFrame(columns=["funding_time", "rate"]) for symbol in symbols}
        funding["AAVEUSDT"] = pd.DataFrame({
            "funding_time": [pd.Timestamp("2021-01-02 04:00:00.002", tz="UTC"),
                             pd.Timestamp("2021-01-02 04:00:01.000", tz="UTC")],
            "rate": [0.01, 0.01],
        })
        signals = {
            start: Signal(start, 14, 7, symbols[:2], symbols[2:], "trade"),
            start + pd.Timedelta(days=1): Signal(start + pd.Timedelta(days=1), 0, 0, (), (), "cash"),
        }
        nav, trades, accounting, liquidated = _simulate(
            trade, mark, funding, signals, start=start, end=end,
            slippage_rate=0.001, leverage=1.0)
        self.assertFalse(liquidated)
        self.assertEqual(len(trades), 8)  # Four entries, then four exits.
        self.assertAlmostEqual(accounting["funding"], -25.0, places=8)
        self.assertAlmostEqual(accounting["taker_fees"], 9.0, places=8)
        self.assertAlmostEqual(accounting["slippage"], 20.0, places=8)
        self.assertAlmostEqual(nav.loc[end], 9_946.0, places=8)

    def test_internal_gap_and_bad_ohlc_fail_closed(self) -> None:
        frame = _constant_bars("2021-01-01", "2021-08-01")
        # One isolated missing bar in >1,000 is within the frozen 99.9% and
        # 8-hour adjacent-opening interpretation; two consecutive are not.
        _audit_bars(frame.drop(frame.index[2]), "one unrelated gap", trade=True)
        with self.assertRaises(DataIntegrityError):
            _audit_bars(frame.drop(frame.index[[2, 3]]), "two-bar gap", trade=True)
        broken = frame.copy()
        broken.loc[broken.index[0], "low"] = 101.0
        with self.assertRaises(DataIntegrityError):
            _audit_bars(broken, "bad ohlc", trade=True)

    def test_leverage_stress_cannot_run_without_successful_1x_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "1x base result"):
            run_reversal_backtest({}, {}, {}, leverage=3.0)

    def test_intrabar_long_low_short_high_liquidates_stress(self) -> None:
        start = pd.Timestamp("2021-01-01", tz="UTC")
        end = pd.Timestamp("2021-01-03", tz="UTC")
        symbols = ("AAVEUSDT", "BANDUSDT", "COMPUSDT", "DASHUSDT")
        trade = {symbol: _constant_bars(str(start), str(end)) for symbol in symbols}
        mark = {symbol: _constant_bars(str(start), str(end)).drop(columns="quote_volume")
                for symbol in symbols}
        for symbol in symbols[:2]:
            mark[symbol].loc[start + pd.Timedelta(hours=4), "low"] = 1.0
        for symbol in symbols[2:]:
            mark[symbol].loc[start + pd.Timedelta(hours=4), "high"] = 200.0
        funding = {symbol: pd.DataFrame(columns=["funding_time", "rate"]) for symbol in symbols}
        signals = {
            start: Signal(start, 14, 7, symbols[:2], symbols[2:], "trade"),
            start + pd.Timedelta(days=1): Signal(start + pd.Timedelta(days=1), 14, 7, symbols[:2], symbols[2:], "trade"),
        }
        nav, trades, accounting, liquidated = _simulate(
            trade, mark, funding, signals, start=start, end=end,
            slippage_rate=0.001, leverage=3.0)
        self.assertTrue(liquidated)
        self.assertEqual(nav.loc[end], 0.0)
        self.assertGreater(accounting["liquidation_fee"], 0)
        self.assertEqual(len(trades), 4)  # No post-liquidation re-entry.

    def test_trade_price_wicks_do_not_trigger_mark_price_liquidation(self) -> None:
        start = pd.Timestamp("2021-01-01", tz="UTC")
        end = pd.Timestamp("2021-01-03", tz="UTC")
        symbols = ("AAVEUSDT", "BANDUSDT", "COMPUSDT", "DASHUSDT")
        trade = {symbol: _constant_bars(str(start), str(end)) for symbol in symbols}
        mark = {symbol: _constant_bars(str(start), str(end)).drop(columns="quote_volume")
                for symbol in symbols}
        for symbol in symbols[:2]:
            trade[symbol].loc[start + pd.Timedelta(hours=4), "low"] = 1.0
        for symbol in symbols[2:]:
            trade[symbol].loc[start + pd.Timedelta(hours=4), "high"] = 200.0
        funding = {symbol: pd.DataFrame(columns=["funding_time", "rate"]) for symbol in symbols}
        signals = {
            start: Signal(start, 14, 7, symbols[:2], symbols[2:], "trade"),
            start + pd.Timedelta(days=1): Signal(start + pd.Timedelta(days=1), 0, 0, (), (), "cash"),
        }
        nav, _, _, liquidated = _simulate(
            trade, mark, funding, signals, start=start, end=end,
            slippage_rate=0.001, leverage=3.0)
        self.assertFalse(liquidated)
        self.assertGreater(nav.loc[end], 0)

    def test_even_1x_short_squeeze_can_trigger_mark_margin(self) -> None:
        start = pd.Timestamp("2021-01-01", tz="UTC")
        end = pd.Timestamp("2021-01-03", tz="UTC")
        symbols = ("AAVEUSDT", "BANDUSDT", "COMPUSDT", "DASHUSDT")
        trade = {symbol: _constant_bars(str(start), str(end)) for symbol in symbols}
        mark = {symbol: _constant_bars(str(start), str(end)).drop(columns="quote_volume")
                for symbol in symbols}
        for symbol in symbols[:2]:
            mark[symbol].loc[start + pd.Timedelta(hours=4), "low"] = 1.0
        for symbol in symbols[2:]:
            mark[symbol].loc[start + pd.Timedelta(hours=4), "high"] = 200.0
        funding = {symbol: pd.DataFrame(columns=["funding_time", "rate"]) for symbol in symbols}
        signals = {
            start: Signal(start, 14, 7, symbols[:2], symbols[2:], "trade"),
            start + pd.Timedelta(days=1): Signal(start + pd.Timedelta(days=1), 0, 0, (), (), "cash"),
        }
        nav, _, accounting, liquidated = _simulate(
            trade, mark, funding, signals, start=start, end=end,
            slippage_rate=0.001, leverage=1.0)
        self.assertTrue(liquidated)
        self.assertEqual(nav.loc[end], 0.0)
        self.assertEqual(accounting["financing"], 0.0)

    def test_08_funding_milliseconds_precede_that_bars_margin_check(self) -> None:
        start = pd.Timestamp("2021-01-01", tz="UTC")
        end = pd.Timestamp("2021-01-03", tz="UTC")
        symbols = ("AAVEUSDT", "BANDUSDT", "COMPUSDT", "DASHUSDT")
        trade = {symbol: _constant_bars(str(start), str(end)) for symbol in symbols}
        mark = {symbol: _constant_bars(str(start), str(end)).drop(columns="quote_volume")
                for symbol in symbols}
        eight = start + pd.Timedelta(hours=8)
        for symbol in symbols[:2]:
            mark[symbol].loc[eight, "low"] = 50.0
        for symbol in symbols[2:]:
            mark[symbol].loc[eight, "high"] = 150.0
        funding = {symbol: pd.DataFrame(columns=["funding_time", "rate"]) for symbol in symbols}
        # Deliberately extreme synthetic rate: without this bar's charge the
        # adverse mark move remains well above maintenance, so event order is
        # directly observable in the liquidation outcome.
        funding["AAVEUSDT"] = pd.DataFrame({
            "funding_time": [pd.Timestamp("2021-01-01 08:00:00.003", tz="UTC")],
            "rate": [2.0],
        })
        signals = {
            start: Signal(start, 14, 7, symbols[:2], symbols[2:], "trade"),
            start + pd.Timedelta(days=1): Signal(start + pd.Timedelta(days=1), 0, 0, (), (), "cash"),
        }
        _, _, accounting, liquidated = _simulate(
            trade, mark, funding, signals, start=start, end=end,
            slippage_rate=0.001, leverage=1.0)
        self.assertTrue(liquidated)
        self.assertAlmostEqual(accounting["funding"], -5_000.0, places=8)

    def test_missing_bar_in_open_position_is_data_failure(self) -> None:
        start = pd.Timestamp("2021-01-01", tz="UTC")
        end = pd.Timestamp("2021-01-03", tz="UTC")
        symbols = ("AAVEUSDT", "BANDUSDT", "COMPUSDT", "DASHUSDT")
        trade = {symbol: _constant_bars(str(start), str(end)) for symbol in symbols}
        mark = {symbol: trade[symbol].drop(columns="quote_volume") for symbol in symbols}
        trade["AAVEUSDT"] = trade["AAVEUSDT"].drop(start + pd.Timedelta(hours=8))
        funding = {symbol: pd.DataFrame(columns=["funding_time", "rate"]) for symbol in symbols}
        signals = {
            start: Signal(start, 14, 7, symbols[:2], symbols[2:], "trade"),
            start + pd.Timedelta(days=1): Signal(start + pd.Timedelta(days=1), 0, 0, (), (), "cash"),
        }
        with self.assertRaisesRegex(DataIntegrityError, "position was open"):
            _simulate(trade, mark, funding, signals, start=start, end=end,
                      slippage_rate=0.001, leverage=1.0)

    def test_daily_metrics_and_block_monte_carlo_are_deterministic(self) -> None:
        dates = pd.date_range("2021-01-01", periods=100, freq="D", tz="UTC")
        returns = pd.Series(np.tile([0.01, -0.005, 0.003, -0.002], 25), index=dates)
        metrics = _metrics(returns)
        self.assertEqual(metrics["days"], 100)
        self.assertGreater(metrics["profit_factor"], 1.0)
        self.assertLessEqual(metrics["max_drawdown"], 0)
        first = _monte_carlo(returns, paths=50, seed=20261102)
        second = _monte_carlo(returns, paths=50, seed=20261102)
        self.assertEqual(first, second)
        self.assertEqual(first["paths"], 50)


if __name__ == "__main__":
    unittest.main()
