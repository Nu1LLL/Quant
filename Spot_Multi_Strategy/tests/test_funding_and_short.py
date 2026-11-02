import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BacktestConfig
from engine import run_backtest
from paper_trader import create_account, process_account
from research.funding_data import attach_funding_rates_to_bars, validate_funding_rates
from strategies import SleeveDefinition


def bars(side):
    return pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC"),
        "open": np.full(5, 100.0),
        "high": np.full(5, 100.1),
        "low": np.full(5, 99.9),
        "close": np.full(5, 100.0),
        "atr": np.full(5, 1.0),
        "funding_rate": [0.0, 0.0, -0.001, 0.0, 0.0],
        "long_entry": [side == "long"] + [False] * 4,
        "long_exit": [False] * 5,
        "short_entry": [side == "short"] + [False] * 4,
        "short_exit": [False] * 5,
    })


def sleeve():
    return SleeveDefinition(
        name="short_engine_safety",
        weight=1.0,
        entry_column="long_entry",
        exit_column="long_exit",
        stop_atr_multiple=2.0,
        short_entry_column="short_entry",
        short_exit_column="short_exit",
        take_profit_r_multiple=2.0,
    )


class FundingAndShortTests(unittest.TestCase):
    def test_direction_caps_apply_separately(self):
        config = BacktestConfig(
            initial_capital=10000, fee_rate=0, slippage_rate=0,
            risk_per_trade=0.05, max_long_notional=3000,
            max_short_notional=3000,
        )
        for side in ("long", "short"):
            result = run_backtest(bars(side), [sleeve()], config)
            trade = result.trades.iloc[0]
            self.assertLessEqual(trade["entry_price"] * trade["quantity"], 3000)

    def test_short_funding_is_separate_from_trading_fees(self):
        result = run_backtest(
            bars("short"), [sleeve()],
            BacktestConfig(initial_capital=10000, fee_rate=0,
                           slippage_rate=0, risk_per_trade=0.01),
        )
        self.assertEqual(result.total_trading_fees, 0)
        self.assertGreater(result.total_funding_fees, 0)
        self.assertAlmostEqual(
            result.trades.iloc[0]["funding_fee"], result.total_funding_fees
        )

    def test_funding_attaches_to_settlement_bar(self):
        candles = bars("long").drop(columns="funding_rate")
        funding = pd.DataFrame({
            "funding_time": [candles.loc[2, "open_time"] + pd.Timedelta(milliseconds=8)],
            "funding_rate": [0.0002],
        })
        attached = attach_funding_rates_to_bars(candles, funding)
        self.assertEqual(attached["funding_rate"].ne(0).sum(), 1)
        self.assertEqual(attached.loc[2, "funding_rate"], 0.0002)

    def test_mixed_iso_funding_times_are_accepted(self):
        funding = pd.DataFrame({
            "funding_time": ["2025-01-01 00:00:00+00:00",
                             "2025-01-01 08:00:00.008000+00:00"],
            "funding_rate": [0.0001, 0.0002],
        })
        self.assertTrue(str(validate_funding_rates(funding)["funding_time"].dtype).endswith("UTC]"))

    def test_short_stop_is_active_on_entry_bar(self):
        candles = bars("short")
        candles.loc[1, "high"] = 105.0
        result = run_backtest(
            candles, [sleeve()],
            BacktestConfig(initial_capital=1000, fee_rate=0,
                           slippage_rate=0, risk_per_trade=0.01),
        )
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "same_bar_stop")

    def test_paper_short_can_exit_at_target(self):
        account = create_account(1000)
        account["last_signal_time"] = "2026-01-01T00:00:00+00:00"
        candle = bars("short").iloc[[1]].copy().reset_index(drop=True)
        candle.loc[0, "short_entry"] = True
        config = BacktestConfig(initial_capital=1000, fee_rate=0,
                                slippage_rate=0, risk_per_trade=0.01)
        account, _, status = process_account(
            account, "BTCUSDT", candle, sleeve(), 100, "1h", config
        )
        self.assertEqual(status, "processed")
        self.assertEqual(account["side"], "short")
        account, _, status = process_account(
            account, "BTCUSDT", candle, sleeve(), 91, "1h", config
        )
        self.assertEqual(status, "take_profit")
        self.assertEqual(account["side"], "flat")


if __name__ == "__main__":
    unittest.main()
