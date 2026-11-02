import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))


from futures_strategy import (
    FuturesRotationConfig,
    generate_futures_targets,
    run_futures_rotation_backtest,
    validate_futures_config
)


def make_frame(start_price, growth, bars=120):
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
            "high": [price * 1.005 for price in closes],
            "low": [price * 0.995 for price in closes],
            "close": closes,
            "volume": [1000.0] * bars
        }
    )


def test_config():
    return FuturesRotationConfig(
        slow_ema_window=10,
        fast_ema_window=5,
        momentum_window=6,
        ema_slope_window=2,
        rebalance_bars=1,
        atr_window=5,
        trailing_stop_atr_multiple=20.0,
        volatility_window=6,
        annual_volatility_target=0.30,
        max_leverage=1.5,
        short_exposure_multiplier=0.8
    )


def empty_funding(symbols):
    columns = [
        "symbol",
        "funding_time",
        "funding_rate",
        "mark_price"
    ]

    return {
        symbol: pd.DataFrame(columns=columns)
        for symbol in symbols
    }


class FuturesStrategyTests(unittest.TestCase):
    def test_leverage_is_hard_capped_at_two(self):
        with self.assertRaises(ValueError):
            validate_futures_config(
                FuturesRotationConfig(max_leverage=2.1)
            )

    def test_future_change_does_not_modify_past_targets(self):
        frames = {
            "BTCUSDT": make_frame(100, 0.003),
            "ETHUSDT": make_frame(100, 0.002)
        }
        config = test_config()
        _, original_targets = generate_futures_targets(frames, config)

        changed_frames = {
            symbol: df.copy()
            for symbol, df in frames.items()
        }
        changed_frames["ETHUSDT"].loc[90:, "close"] *= 4
        changed_frames["ETHUSDT"].loc[90:, "open"] *= 4
        changed_frames["ETHUSDT"].loc[90:, "high"] *= 4
        changed_frames["ETHUSDT"].loc[90:, "low"] *= 4
        _, changed_targets = generate_futures_targets(
            changed_frames,
            config
        )

        pd.testing.assert_frame_equal(
            original_targets.iloc[:90],
            changed_targets.iloc[:90]
        )

    def test_short_position_profits_in_falling_market(self):
        frames = {
            "BTCUSDT": make_frame(100, -0.003),
            "ETHUSDT": make_frame(100, -0.002)
        }
        result = run_futures_rotation_backtest(
            original_frames=frames,
            funding_frames=empty_funding(frames),
            config=test_config(),
            initial_capital=1000,
            fee_rate=0,
            slippage_rate=0
        )

        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["direction"] == "short").all())
        self.assertGreater(result.final_value, 1000)

    def test_positive_funding_cost_reduces_long_result(self):
        frames = {
            "BTCUSDT": make_frame(100, 0.003),
            "ETHUSDT": make_frame(100, 0.002)
        }
        funding = empty_funding(frames)
        funding["BTCUSDT"] = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"] * 8,
                "funding_time": frames["BTCUSDT"][
                    "open_time"
                ].iloc[40:48].tolist(),
                "funding_rate": [0.001] * 8,
                "mark_price": frames["BTCUSDT"][
                    "close"
                ].iloc[40:48].tolist()
            }
        )

        free_result = run_futures_rotation_backtest(
            original_frames=frames,
            funding_frames=empty_funding(frames),
            config=test_config(),
            initial_capital=1000,
            fee_rate=0,
            slippage_rate=0
        )
        funded_result = run_futures_rotation_backtest(
            original_frames=frames,
            funding_frames=funding,
            config=test_config(),
            initial_capital=1000,
            fee_rate=0,
            slippage_rate=0
        )

        self.assertLess(
            funded_result.final_value,
            free_result.final_value
        )
        self.assertLess(
            funded_result.trades["funding_cash_flow"].sum(),
            0
        )
        self.assertGreater(funded_result.total_funding_fees, 0)
        self.assertEqual(funded_result.total_trading_fees, 0)
        self.assertAlmostEqual(
            funded_result.total_fees,
            funded_result.total_trading_fees
            + funded_result.total_funding_fees
        )

    def test_signal_executes_at_next_bar_open(self):
        frames = {
            "BTCUSDT": make_frame(100, 0.003),
            "ETHUSDT": make_frame(100, 0.002)
        }
        config = test_config()
        _, targets = generate_futures_targets(frames, config)
        first_signal_position = next(
            i
            for i, row in targets.iterrows()
            if (
                pd.notna(row["target_symbol"])
                and
                int(row["target_direction"]) != 0
                and
                float(row["target_leverage"]) > 0
            )
        )
        result = run_futures_rotation_backtest(
            original_frames=frames,
            funding_frames=empty_funding(frames),
            config=config,
            fee_rate=0,
            slippage_rate=0
        )
        expected_time = frames["BTCUSDT"].iloc[
            first_signal_position + 1
        ]["open_time"]

        self.assertEqual(
            result.trades.iloc[0]["entry_time"],
            expected_time
        )

    def test_costs_reduce_final_value(self):
        frames = {
            "BTCUSDT": make_frame(100, 0.003),
            "ETHUSDT": make_frame(100, 0.002)
        }
        funding = empty_funding(frames)
        free_result = run_futures_rotation_backtest(
            frames,
            funding,
            config=test_config(),
            fee_rate=0,
            slippage_rate=0
        )
        costly_result = run_futures_rotation_backtest(
            frames,
            funding,
            config=test_config(),
            fee_rate=0.001,
            slippage_rate=0.001
        )

        self.assertLess(costly_result.final_value, free_result.final_value)
        self.assertGreater(costly_result.total_trading_fees, 0)
        self.assertEqual(costly_result.total_funding_fees, 0)
        self.assertAlmostEqual(
            costly_result.total_fees,
            costly_result.total_trading_fees
        )


if __name__ == "__main__":
    unittest.main()
