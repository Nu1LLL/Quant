import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))


from config import BacktestConfig, StrategyConfig, validate_config
from data import repair_small_time_gaps, validate_ohlcv
from engine import run_backtest
from indicators import calculate_adx
from metrics import (
    calculate_monthly_report,
    calculate_monthly_summary,
    calculate_strategy_report,
    evaluate_research_gates
)
from strategies import SleeveDefinition, generate_signals


def make_ohlcv(length=500, interval="4h"):
    # 生成固定、可重复的测试K线
    random_generator = np.random.default_rng(20260822)

    time_index = pd.date_range(
        "2024-01-01",
        periods=length,
        freq=interval,
        tz="UTC"
    )

    trend = np.linspace(100, 150, length)
    cycle = 5 * np.sin(np.arange(length) / 12)
    noise = random_generator.normal(0, 0.5, length)
    close = trend + cycle + noise

    open_price = np.r_[close[0], close[:-1]]
    high = np.maximum(open_price, close) + 1
    low = np.minimum(open_price, close) - 1

    return pd.DataFrame(
        {
            "open_time": time_index,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(length, 1000.0)
        }
    )


class DataTests(unittest.TestCase):
    def test_validate_ohlcv_accepts_continuous_data(self):
        df = make_ohlcv(length=50)
        result = validate_ohlcv(df, interval="4h")
        self.assertEqual(len(result), 50)

    def test_validate_ohlcv_rejects_duplicate_time(self):
        df = make_ohlcv(length=50)
        df.loc[10, "open_time"] = df.loc[9, "open_time"]

        with self.assertRaises(ValueError):
            validate_ohlcv(df, interval="4h")

    def test_validate_ohlcv_rejects_time_gap(self):
        df = make_ohlcv(length=50).drop(index=10)

        with self.assertRaises(ValueError):
            validate_ohlcv(df, interval="4h")

    def test_repair_small_time_gap(self):
        df = make_ohlcv(length=50).drop(index=10)
        repaired_df = repair_small_time_gaps(
            df,
            interval="4h"
        )

        self.assertEqual(len(repaired_df), 50)
        self.assertEqual(int(repaired_df["is_synthetic"].sum()), 1)
        repaired_row = repaired_df.iloc[10]
        previous_close = repaired_df.iloc[9]["close"]
        self.assertEqual(repaired_row["close"], previous_close)
        self.assertEqual(repaired_row["volume"], 0)

    def test_repair_refuses_large_time_gap(self):
        df = make_ohlcv(length=50).drop(index=range(10, 20))

        with self.assertRaises(ValueError):
            repair_small_time_gaps(
                df,
                interval="4h",
                maximum_consecutive_missing=6
            )


class SignalTests(unittest.TestCase):
    def test_signals_do_not_modify_input(self):
        df = make_ohlcv()
        original_columns = list(df.columns)

        signal_df, sleeves = generate_signals(
            df,
            StrategyConfig()
        )

        self.assertEqual(list(df.columns), original_columns)
        self.assertEqual(len(sleeves), 2)
        self.assertIn("trend_entry", signal_df.columns)
        self.assertIn("pullback_entry", signal_df.columns)

    def test_future_prices_do_not_change_past_signals(self):
        df = make_ohlcv()
        first_signal_df, _ = generate_signals(df)

        changed_df = df.copy()
        changed_df.loc[400:, "close"] *= 3
        changed_df.loc[400:, "high"] = np.maximum(
            changed_df.loc[400:, "high"],
            changed_df.loc[400:, "close"] + 1
        )

        second_signal_df, _ = generate_signals(changed_df)

        columns = [
            "trend_entry",
            "trend_exit",
            "pullback_entry",
            "pullback_exit",
            "range_entry",
            "range_exit",
            "trend_regime",
            "range_regime"
        ]

        pd.testing.assert_frame_equal(
            first_signal_df.loc[:399, columns],
            second_signal_df.loc[:399, columns]
        )

    def test_one_strategy_can_be_disabled_with_zero_weight(self):
        df = make_ohlcv()
        _, sleeves = generate_signals(
            df,
            StrategyConfig(
                trend_weight=1.0,
                pullback_weight=0.0
            )
        )

        self.assertEqual(len(sleeves), 1)
        self.assertEqual(sleeves[0].name, "trend_breakout")

    def test_regime_strategy_entries_are_mutually_exclusive(self):
        df = make_ohlcv(length=800)
        strategy_config = StrategyConfig(
            trend_weight=0.60,
            pullback_weight=0.0,
            range_weight=0.40,
            use_regime_filter=True
        )
        signal_df, sleeves = generate_signals(
            df,
            strategy_config
        )

        simultaneous_entries = (
            signal_df["trend_entry"]
            &
            signal_df["range_entry"]
        )

        self.assertFalse(simultaneous_entries.any())
        self.assertEqual(len(sleeves), 2)
        self.assertEqual(
            {sleeve.name for sleeve in sleeves},
            {"trend_breakout", "range_mean_reversion"}
        )

    def test_range_strategy_requires_regime_filter(self):
        strategy_config = StrategyConfig(
            trend_weight=0.60,
            pullback_weight=0.0,
            range_weight=0.40,
            use_regime_filter=False
        )

        with self.assertRaises(ValueError):
            validate_config(
                strategy_config,
                BacktestConfig()
            )

    def test_adx_columns_are_valid_after_warmup(self):
        df = make_ohlcv(length=200)
        adx_data = calculate_adx(df, window=14).dropna()

        self.assertFalse(adx_data.empty)
        self.assertTrue((adx_data["adx"] >= 0).all())
        self.assertTrue((adx_data["adx"] <= 100).all())


class EngineTests(unittest.TestCase):
    def test_entry_executes_at_next_open_with_fees(self):
        df = make_ohlcv(length=5)
        df["atr"] = 2.0
        df["entry"] = [True, False, False, False, False]
        df["exit"] = [False, True, False, False, False]

        sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=10.0,
            trailing_atr_multiple=None
        )

        config = BacktestConfig(
            initial_capital=1000,
            fee_rate=0.001,
            slippage_rate=0,
            risk_per_trade=0.01,
            max_sleeve_exposure=1.0
        )

        result = run_backtest(df, [sleeve], config)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertAlmostEqual(
            trade["entry_price"],
            df.iloc[1]["open"]
        )
        self.assertAlmostEqual(
            trade["exit_price"],
            df.iloc[2]["open"]
        )
        self.assertGreater(result.total_fees, 0)

    def test_execution_delay_uses_later_open(self):
        df = make_ohlcv(length=6)
        df["atr"] = 2.0
        df["entry"] = [True, False, False, False, False, False]
        df["exit"] = [False, True, False, False, False, False]

        sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=10.0,
            trailing_atr_multiple=None
        )
        result = run_backtest(
            df,
            [sleeve],
            BacktestConfig(
                fee_rate=0,
                slippage_rate=0,
                execution_delay_bars=2
            )
        )

        trade = result.trades.iloc[0]
        self.assertEqual(trade["entry_price"], df.iloc[2]["open"])
        self.assertEqual(trade["exit_price"], df.iloc[3]["open"])

    def test_trade_records_entry_account_value(self):
        df = make_ohlcv(length=5)
        df["atr"] = 2.0
        df["entry"] = [True, False, False, False, False]
        df["exit"] = [False, True, False, False, False]
        sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=10.0,
            trailing_atr_multiple=None
        )

        result = run_backtest(
            df,
            [sleeve],
            BacktestConfig(initial_capital=1000)
        )

        self.assertAlmostEqual(
            result.trades.iloc[0]["entry_account_value"],
            1000.0
        )

    def test_engine_never_uses_leverage(self):
        df = make_ohlcv(length=10)
        df["atr"] = 0.01
        df["entry"] = [True] + [False] * 9
        df["exit"] = [False] * 9 + [True]

        sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=2.0,
            trailing_atr_multiple=None
        )

        result = run_backtest(
            df,
            [sleeve],
            BacktestConfig(
                initial_capital=1000,
                risk_per_trade=0.05,
                max_sleeve_exposure=1.0
            )
        )

        trade = result.trades.iloc[0]
        gross_entry_value = (
            trade["entry_price"] * trade["quantity"]
        )

        self.assertLessEqual(gross_entry_value, 1000.0)

    def test_stop_is_active_on_entry_bar(self):
        df = make_ohlcv(length=4)
        df["atr"] = 2.0
        df["entry"] = [True, False, False, False]
        df["exit"] = [False, False, False, False]

        # 第2根K线开盘买入以后，盘中最低价跌破止损价
        entry_open = float(df.iloc[1]["open"])
        df.loc[1, "low"] = entry_open - 3.0

        sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=1.0,
            trailing_atr_multiple=None
        )

        result = run_backtest(
            df,
            [sleeve],
            BacktestConfig(
                initial_capital=1000,
                fee_rate=0,
                slippage_rate=0,
                risk_per_trade=0.01
            )
        )

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(
            result.trades.iloc[0]["exit_reason"],
            "same_bar_stop"
        )

    def test_exit_and_entry_cannot_happen_on_same_bar(self):
        df = make_ohlcv(length=5)
        df["atr"] = 2.0
        df["entry"] = [True, True, False, False, False]
        df["exit"] = [False, True, False, False, False]

        sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=10.0,
            trailing_atr_multiple=None
        )

        result = run_backtest(
            df,
            [sleeve],
            BacktestConfig(
                initial_capital=1000,
                fee_rate=0,
                slippage_rate=0,
                risk_per_trade=0.01
            )
        )

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(
            result.trades.iloc[0]["exit_reason"],
            "signal"
        )


class MetricsTests(unittest.TestCase):
    def test_strategy_and_monthly_reports_are_generated(self):
        df = make_ohlcv(length=50)
        df["atr"] = 2.0
        df["entry"] = [True] + [False] * 49
        df["exit"] = [False, True] + [False] * 48

        sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=10.0,
            trailing_atr_multiple=None
        )

        result = run_backtest(df, [sleeve])
        strategy_report = calculate_strategy_report(result)
        monthly_report = calculate_monthly_report(result)
        monthly_summary = calculate_monthly_summary(monthly_report)

        self.assertEqual(strategy_report.iloc[0]["strategy"], "test")
        self.assertEqual(monthly_summary["month_count"], 1)

    def test_research_gate_rejects_weak_sample(self):
        weak_metrics = {
            "total_return": -0.01,
            "trade_count": 10,
            "profit_factor": 0.9,
            "sharpe_ratio": -0.2,
            "max_drawdown": -0.25
        }

        gate_result = evaluate_research_gates(weak_metrics)
        self.assertFalse(gate_result["passed"])


if __name__ == "__main__":
    unittest.main()
