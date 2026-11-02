import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))


from rotation_strategy import (
    RotationConfig,
    calculate_rotation_exposure,
    generate_rotation_targets,
    run_rotation_backtest,
    validate_rotation_config
)
from rotation_research import make_research_checks


def make_frame(start_price, growth, bars=80):
    times = pd.date_range(
        "2025-01-01",
        periods=bars,
        freq="4h",
        tz="UTC"
    )
    closes = [
        start_price * (1 + growth) ** i
        for i in range(bars)
    ]

    return pd.DataFrame(
        {
            "open_time": times,
            "open": [price * 0.999 for price in closes],
            "high": [price * 1.01 for price in closes],
            "low": [price * 0.99 for price in closes],
            "close": closes,
            "volume": [1000.0] * bars
        }
    )


def test_config():
    return RotationConfig(
        slow_ema_window=8,
        fast_ema_window=4,
        momentum_window=5,
        ema_slope_window=2,
        rebalance_bars=1,
        exposure=0.35
    )


class RotationStrategyTests(unittest.TestCase):
    def setUp(self):
        self.frames = {
            "BTCUSDT": make_frame(100, 0.004),
            "ETHUSDT": make_frame(100, 0.002)
        }
        self.config = test_config()

    def test_fast_ema_must_be_shorter_than_slow_ema(self):
        with self.assertRaises(ValueError):
            validate_rotation_config(
                RotationConfig(
                    slow_ema_window=10,
                    fast_ema_window=10
                )
            )

    def test_future_change_does_not_modify_past_targets(self):
        _, original_targets = generate_rotation_targets(
            self.frames,
            self.config
        )

        changed_frames = {
            symbol: df.copy()
            for symbol, df in self.frames.items()
        }
        changed_frames["ETHUSDT"].loc[60:, "close"] *= 5
        changed_frames["ETHUSDT"].loc[60:, "open"] *= 5
        changed_frames["ETHUSDT"].loc[60:, "high"] *= 5
        changed_frames["ETHUSDT"].loc[60:, "low"] *= 5

        _, changed_targets = generate_rotation_targets(
            changed_frames,
            self.config
        )

        pd.testing.assert_series_equal(
            original_targets.iloc[:60],
            changed_targets.iloc[:60]
        )

    def test_signal_executes_at_next_bar_open(self):
        _, targets = generate_rotation_targets(
            self.frames,
            self.config
        )
        first_signal_position = next(
            i
            for i, value in enumerate(targets)
            if value is not None
        )

        result = run_rotation_backtest(
            self.frames,
            config=self.config,
            fee_rate=0,
            slippage_rate=0
        )

        expected_time = self.frames["BTCUSDT"].iloc[
            first_signal_position + 1
        ]["open_time"]

        self.assertEqual(
            result.trades.iloc[0]["entry_time"],
            expected_time
        )

    def test_exposure_keeps_most_capital_in_cash(self):
        result = run_rotation_backtest(
            self.frames,
            config=self.config,
            initial_capital=1000,
            fee_rate=0,
            slippage_rate=0
        )

        invested_rows = result.equity_curve[
            result.equity_curve["held_symbol"].notna()
        ]

        self.assertFalse(invested_rows.empty)
        self.assertGreaterEqual(invested_rows["cash"].min(), 649.99)

    def test_broad_market_exposure_uses_more_capital(self):
        adaptive_config = RotationConfig(
            slow_ema_window=8,
            fast_ema_window=4,
            momentum_window=5,
            ema_slope_window=2,
            rebalance_bars=1,
            exposure=0.35,
            broad_market_exposure=0.40
        )

        fixed_result = run_rotation_backtest(
            self.frames,
            config=self.config,
            initial_capital=1000,
            fee_rate=0,
            slippage_rate=0
        )
        adaptive_result = run_rotation_backtest(
            self.frames,
            config=adaptive_config,
            initial_capital=1000,
            fee_rate=0,
            slippage_rate=0
        )

        fixed_cash = fixed_result.equity_curve[
            fixed_result.equity_curve["held_symbol"].notna()
        ]["cash"].min()
        adaptive_cash = adaptive_result.equity_curve[
            adaptive_result.equity_curve["held_symbol"].notna()
        ]["cash"].min()

        self.assertLess(adaptive_cash, fixed_cash)

    def test_broad_market_exposure_cannot_be_lower(self):
        with self.assertRaises(ValueError):
            validate_rotation_config(
                RotationConfig(
                    exposure=0.40,
                    broad_market_exposure=0.33
                )
            )

    def test_volatility_parameters_must_be_supplied_together(self):
        with self.assertRaises(ValueError):
            validate_rotation_config(
                RotationConfig(
                    volatility_window=75
                )
            )

    def test_volatility_target_reduces_position_size(self):
        volatility_config = RotationConfig(
            slow_ema_window=8,
            fast_ema_window=4,
            momentum_window=5,
            ema_slope_window=2,
            rebalance_bars=1,
            exposure=1.0,
            broad_market_exposure=1.0,
            volatility_window=10,
            target_volatility=0.24,
            minimum_exposure=0.10
        )
        frames = {
            "BTCUSDT": pd.DataFrame(
                {
                    "eligible": [True],
                    "realized_volatility": [0.48]
                }
            ),
            "ETHUSDT": pd.DataFrame(
                {
                    "eligible": [True],
                    "realized_volatility": [0.30]
                }
            )
        }

        exposure = calculate_rotation_exposure(
            frames=frames,
            desired_symbol="BTCUSDT",
            signal_position=0,
            config=volatility_config
        )

        self.assertAlmostEqual(exposure, 0.50)

    def test_costs_reduce_final_value(self):
        free_result = run_rotation_backtest(
            self.frames,
            config=self.config,
            fee_rate=0,
            slippage_rate=0
        )
        costly_result = run_rotation_backtest(
            self.frames,
            config=self.config,
            fee_rate=0.002,
            slippage_rate=0.001
        )

        self.assertLess(
            costly_result.final_value,
            free_result.final_value
        )
        self.assertGreater(costly_result.total_trading_fees, 0)
        self.assertEqual(costly_result.total_funding_fees, 0)
        self.assertAlmostEqual(
            costly_result.total_fees,
            costly_result.total_trading_fees
        )

    def test_recent_metrics_cannot_be_hidden_by_old_bull_market(self):
        strong_full_metrics = {
            "annualized_return": 0.40,
            "max_drawdown": -0.10,
            "sharpe_ratio": 2.0,
            "total_return": 4.0
        }
        weak_recent_metrics = {
            "annualized_return": 0.08,
            "max_drawdown": -0.14,
            "sharpe_ratio": 0.70,
            "total_return": 0.40
        }
        rolling_report = pd.DataFrame(
            {
                "annualized_return": [0.05, 0.10],
                "max_drawdown": [-0.10, -0.12],
                "sharpe_ratio": [0.80, 1.10]
            }
        )

        checks = make_research_checks(
            base_metrics=strong_full_metrics,
            recent_metrics=weak_recent_metrics,
            recent_double_cost_metrics=weak_recent_metrics,
            recent_delayed_metrics=weak_recent_metrics,
            recent_rolling_report=rolling_report,
            neighborhood_summary={"passed_ratio": 0.0}
        )

        self.assertTrue(checks["完整历史仍然盈利"])
        self.assertFalse(
            checks["近年独立起算年化位于20%至25%"]
        )
        self.assertFalse(
            checks["近年独立起算Sharpe不低于1"]
        )


if __name__ == "__main__":
    unittest.main()
