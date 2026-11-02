import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))


from config import BacktestConfig, StrategyConfig
from paper_trader import (
    PaperDataGapError,
    calculate_config_hash,
    create_account,
    process_account
)
from strategies import SleeveDefinition


def make_signal_df(times, entries=None, exits=None, lows=None):
    length = len(times)
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(times, utc=True),
            "open": [100.0] * length,
            "high": [112.0] * length,
            "low": lows or [99.0] * length,
            "close": [110.0] * length,
            "atr": [2.0] * length,
            "entry": entries or [False] * length,
            "exit": exits or [False] * length
        }
    )


class PaperTraderTests(unittest.TestCase):
    def setUp(self):
        self.sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=2.0,
            trailing_atr_multiple=3.0
        )
        self.config = BacktestConfig(
            initial_capital=1000,
            fee_rate=0,
            slippage_rate=0,
            risk_per_trade=0.01
        )

    def test_config_hash_is_repeatable(self):
        first_hash = calculate_config_hash(
            StrategyConfig(),
            self.config
        )
        second_hash = calculate_config_hash(
            StrategyConfig(),
            self.config
        )

        self.assertEqual(first_hash, second_hash)

    def test_first_run_only_initializes_state(self):
        account, events, status = process_account(
            original_account=create_account(1000),
            symbol="BTCUSDT",
            signal_df=make_signal_df(
                ["2026-01-01 00:00:00+00:00"],
                entries=[True]
            ),
            sleeve=self.sleeve,
            current_price=100,
            interval="4h",
            backtest_config=self.config
        )

        self.assertEqual(status, "initialized")
        self.assertEqual(events, [])
        self.assertEqual(account["quantity"], 0)

    def test_next_bar_can_buy_only_once(self):
        account = create_account(1000)
        account["last_signal_time"] = "2026-01-01T00:00:00+00:00"
        signal_df = make_signal_df(
            ["2026-01-01 04:00:00+00:00"],
            entries=[True]
        )

        account, events, status = process_account(
            account,
            "BTCUSDT",
            signal_df,
            self.sleeve,
            current_price=100,
            interval="4h",
            backtest_config=self.config
        )

        self.assertEqual(status, "processed")
        self.assertEqual(len(events), 1)
        self.assertGreater(account["quantity"], 0)
        self.assertEqual(account["highest_close"], 100)
        self.assertEqual(account["stop_price"], 96)

        duplicate_account, duplicate_events, duplicate_status = (
            process_account(
                account,
                "BTCUSDT",
                signal_df,
                self.sleeve,
                current_price=101,
                interval="4h",
                backtest_config=self.config
            )
        )

        self.assertEqual(duplicate_status, "duplicate")
        self.assertEqual(duplicate_events, [])
        self.assertEqual(
            duplicate_account["quantity"],
            account["quantity"]
        )

    def test_missed_bar_stops_processing(self):
        account = create_account(1000)
        account["last_signal_time"] = "2026-01-01T00:00:00+00:00"

        with self.assertRaises(PaperDataGapError):
            process_account(
                account,
                "BTCUSDT",
                make_signal_df(
                    ["2026-01-01 08:00:00+00:00"]
                ),
                self.sleeve,
                current_price=100,
                interval="4h",
                backtest_config=self.config
            )

    def test_duplicate_run_can_execute_observed_stop(self):
        account = create_account(1000)
        account.update(
            {
                "cash": 0.0,
                "quantity": 10.0,
                "entry_price": 100.0,
                "entry_time": "2026-01-01T00:05:00+00:00",
                "entry_account_value": 1000.0,
                "stop_price": 96.0,
                "highest_close": 100.0,
                "last_signal_time": "2026-01-01T00:00:00+00:00"
            }
        )

        account, events, status = process_account(
            account,
            "BTCUSDT",
            make_signal_df(
                ["2026-01-01 00:00:00+00:00"]
            ),
            self.sleeve,
            current_price=95,
            interval="4h",
            backtest_config=self.config
        )

        self.assertEqual(status, "stop")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["exit_reason"], "observed_price_stop")
        self.assertEqual(account["quantity"], 0)
        self.assertEqual(account["cash"], 950)


if __name__ == "__main__":
    unittest.main()
