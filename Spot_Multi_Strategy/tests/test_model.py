import io
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))


from config import BacktestConfig, StrategyConfig, validate_config
from config_io import load_strategy_config
from data import (
    _mixed_epoch_to_utc,
    download_monthly_archive_klines,
    repair_small_time_gaps,
    validate_ohlcv
)
from engine import run_backtest
from indicators import calculate_adx
from metrics import (
    calculate_metrics,
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

    def test_1m_gap_policy_uses_two_hour_duration(self):
        df = make_ohlcv(length=300, interval="1min")
        repaired = repair_small_time_gaps(
            df.drop(index=range(50, 130)), interval="1m"
        )
        self.assertEqual(int(repaired["is_synthetic"].sum()), 80)

        with self.assertRaises(ValueError):
            repair_small_time_gaps(
                df.drop(index=range(50, 171)), interval="1m"
            )

    def test_monthly_archive_accepts_microsecond_timestamps(self):
        buffer = io.BytesIO()
        rows = [
            [
                1735689600000000, 100, 101, 99, 100, 10,
                1735689659999999, 1000, 5, 4, 400, 0
            ],
            [
                1735689660000000, 100, 102, 100, 101, 20,
                1735689719999999, 2000, 6, 8, 800, 0
            ]
        ]
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "BTCUSDT-1m-2025-01.csv",
                "\n".join(",".join(map(str, row)) for row in rows)
            )

        class ArchiveResponse:
            content = buffer.getvalue()

            def raise_for_status(self):
                return None

        class ArchiveSession:
            def get(self, *args, **kwargs):
                return ArchiveResponse()

        result = download_monthly_archive_klines(
            "BTCUSDT",
            "1m",
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:02:00Z",
            session=ArchiveSession()
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[1]["close"], 101)

    def test_mixed_epoch_units_are_parsed_per_row(self):
        values = pd.Series([1704067200000, 1735689600000000])
        result = _mixed_epoch_to_utc(values)
        self.assertEqual(result.iloc[0], pd.Timestamp("2024-01-01", tz="UTC"))
        self.assertEqual(result.iloc[1], pd.Timestamp("2025-01-01", tz="UTC"))


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

    def test_profit_first_config_uses_only_minimal_structural_trend(self):
        strategy_config = load_strategy_config(
            PROJECT_FOLDER / "configs" / "profit_first_simplified.json"
        )

        signal_df, sleeves = generate_signals(
            make_ohlcv(length=500), strategy_config
        )

        self.assertEqual(len(sleeves), 1)
        sleeve = sleeves[0]
        self.assertEqual(sleeve.name, "profit_first_structure_trend")
        self.assertEqual(sleeve.long_stop_column, "trend_structural_stop")
        self.assertIsNone(sleeve.stop_atr_multiple)
        self.assertIsNone(sleeve.trailing_atr_multiple)

        expected_columns = {
            "ema",
            "ema_rising",
            "entry_high",
            "exit_low",
            "trend_structural_stop",
            "trend_entry",
            "trend_exit"
        }
        self.assertTrue(expected_columns.issubset(signal_df.columns))
        unused_indicator_columns = {
            "atr",
            "rsi",
            "macd",
            "macd_signal",
            "bollinger_upper",
            "bollinger_lower",
            "adx"
        }
        self.assertTrue(unused_indicator_columns.isdisjoint(signal_df.columns))

    def test_zero_structural_stop_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "趋势结构止损周期必须大于0"):
            validate_config(
                StrategyConfig(trend_structural_stop_window=0),
                BacktestConfig()
            )

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

    def test_slippage_is_separate_for_long_and_short(self):
        for side in ("long", "short"):
            with self.subTest(side=side):
                df = make_ohlcv(length=5)
                # Keep the path flat so this test isolates accounting rather
                # than triggering the short sleeve's intrabar stop first.
                df[["open", "high", "low", "close"]] = 100.0
                df["atr"] = 2.0
                df["long_entry"] = False
                df["long_exit"] = False
                df["short_entry"] = False
                df["short_exit"] = False
                df.loc[0, f"{side}_entry"] = True
                df.loc[1, f"{side}_exit"] = True

                sleeve = SleeveDefinition(
                    name="test",
                    weight=1.0,
                    entry_column="long_entry",
                    exit_column="long_exit",
                    stop_atr_multiple=10.0,
                    short_entry_column="short_entry",
                    short_exit_column="short_exit"
                )
                result = run_backtest(
                    df,
                    [sleeve],
                    BacktestConfig(
                        initial_capital=1000,
                        fee_rate=0.001,
                        short_fee_rate=0.0005,
                        slippage_rate=0.01,
                        risk_per_trade=0.05
                    )
                )

                trade = result.trades.iloc[0]
                raw_entry_price = float(df.iloc[1]["open"])
                raw_exit_price = float(df.iloc[2]["open"])
                quantity = trade["quantity"]
                market_direction = 1 if side == "long" else -1

                self.assertAlmostEqual(
                    trade["entry_slippage"],
                    abs(trade["entry_price"] - raw_entry_price) * quantity
                )
                self.assertAlmostEqual(
                    trade["exit_slippage"],
                    abs(trade["exit_price"] - raw_exit_price) * quantity
                )
                self.assertAlmostEqual(
                    trade["slippage_cost"],
                    trade["entry_slippage"] + trade["exit_slippage"]
                )
                self.assertAlmostEqual(
                    trade["gross_market_profit"],
                    market_direction
                    * (raw_exit_price - raw_entry_price)
                    * quantity
                )
                self.assertAlmostEqual(
                    trade["gross_profit"],
                    trade["gross_market_profit"] - trade["slippage_cost"]
                )
                self.assertAlmostEqual(
                    trade["net_profit"],
                    trade["gross_market_profit"]
                    - trade["slippage_cost"]
                    - trade["trading_fee"]
                    - trade["funding_fee"]
                )
                self.assertAlmostEqual(
                    result.total_slippage_cost,
                    trade["slippage_cost"]
                )
                self.assertAlmostEqual(
                    result.total_fees,
                    result.total_trading_fees + result.total_funding_fees
                )

    def test_stop_fraction_limits_use_slippage_adjusted_entry(self):
        df = make_ohlcv(length=4)
        df["long_entry"] = [True, False, False, False]
        df["long_exit"] = [False, True, False, False]
        raw_entry_price = float(df.iloc[1]["open"])
        df["structural_stop"] = raw_entry_price * 0.99

        maximum_limited_sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="long_entry",
            exit_column="long_exit",
            stop_atr_multiple=None,
            long_stop_column="structural_stop",
            maximum_stop_fraction=0.015
        )
        rejected_for_slippage = run_backtest(
            df,
            [maximum_limited_sleeve],
            BacktestConfig(fee_rate=0, slippage_rate=0.01)
        )
        accepted_without_slippage = run_backtest(
            df,
            [maximum_limited_sleeve],
            BacktestConfig(fee_rate=0, slippage_rate=0)
        )

        self.assertTrue(rejected_for_slippage.trades.empty)
        self.assertEqual(len(accepted_without_slippage.trades), 1)

        df["structural_stop"] = raw_entry_price * 0.9995
        minimum_limited_sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="long_entry",
            exit_column="long_exit",
            stop_atr_multiple=None,
            long_stop_column="structural_stop",
            minimum_stop_fraction=0.001
        )
        rejected_as_too_tight = run_backtest(
            df,
            [minimum_limited_sleeve],
            BacktestConfig(fee_rate=0, slippage_rate=0)
        )
        self.assertTrue(rejected_as_too_tight.trades.empty)

    def test_signal_exit_at_open_precedes_current_bar_high_low(self):
        df = make_ohlcv(length=4)
        df[["open", "high", "low", "close"]] = 100.0
        df.loc[2, ["open", "close"]] = 101.0
        df.loc[2, "high"] = 150.0
        df["atr"] = 1.0
        df["entry"] = [True, False, False, False]
        df["exit"] = [False, True, False, False]
        sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=10.0,
            take_profit_r_multiple=2.0,
        )

        result = run_backtest(
            df,
            [sleeve],
            BacktestConfig(fee_rate=0, slippage_rate=0),
        )

        trade = result.trades.iloc[0]
        self.assertEqual(trade["exit_reason"], "signal")
        self.assertEqual(trade["exit_price"], 101.0)

    def test_two_r_target_is_net_of_fees_and_slippage(self):
        df = make_ohlcv(length=4)
        df[["open", "high", "low", "close"]] = 100.0
        df.loc[2, "high"] = 150.0
        df["entry"] = [True, False, False, False]
        df["exit"] = False
        df["structural_stop"] = 99.0
        sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=None,
            long_stop_column="structural_stop",
            take_profit_r_multiple=2.0,
            cost_adjusted_target=True,
        )
        risk_fraction = 0.01

        result = run_backtest(
            df,
            [sleeve],
            BacktestConfig(
                initial_capital=1000.0,
                fee_rate=0.001,
                slippage_rate=0.01,
                risk_per_trade=risk_fraction,
                max_long_notional=1000.0,
            ),
        )

        trade = result.trades.iloc[0]
        self.assertEqual(trade["exit_reason"], "take_profit")
        self.assertAlmostEqual(
            trade["net_profit"],
            2.0 * 1000.0 * risk_fraction,
            places=7,
        )

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

    def test_max_holding_bars_forces_timeout_exit(self):
        df = make_ohlcv(length=6)
        df["atr"] = 1.0
        df["entry"] = [True, False, False, False, False, False]
        df["exit"] = False
        sleeve = SleeveDefinition(
            name="test",
            weight=1.0,
            entry_column="entry",
            exit_column="exit",
            stop_atr_multiple=10.0,
            max_holding_bars=2
        )
        result = run_backtest(
            df, [sleeve], BacktestConfig(fee_rate=0, slippage_rate=0)
        )
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "timeout")
        self.assertEqual(
            result.trades.iloc[0]["exit_time"], df.iloc[3]["open_time"]
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
        metrics = calculate_metrics(result)

        self.assertEqual(strategy_report.iloc[0]["strategy"], "test")
        self.assertEqual(monthly_summary["month_count"], 1)
        self.assertAlmostEqual(
            1 + metrics["annualized_return"],
            (1 + metrics["monthly_compound_return"]) ** 12,
            places=12
        )
        trade = result.trades.iloc[0]
        strategy = strategy_report.iloc[0]
        self.assertAlmostEqual(
            metrics["slippage_costs"], result.total_slippage_cost
        )
        self.assertAlmostEqual(
            metrics["total_costs"],
            metrics["total_fees"] + metrics["slippage_costs"]
        )
        self.assertAlmostEqual(
            strategy["gross_market_profit"],
            trade["gross_market_profit"]
        )
        self.assertAlmostEqual(
            strategy["slippage_costs"], trade["slippage_cost"]
        )
        self.assertAlmostEqual(
            strategy["total_fees"],
            strategy["trading_fees"] + strategy["funding_fees"]
        )
        self.assertAlmostEqual(
            strategy["total_costs"],
            strategy["total_fees"] + strategy["slippage_costs"]
        )

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
