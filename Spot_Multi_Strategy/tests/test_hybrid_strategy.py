import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))


from hybrid_strategy import (
    HybridConfig,
    generate_hybrid_targets,
    run_hybrid_backtest,
    validate_hybrid_config
)


def make_frame(start_price, growth, bars=140):
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
            "high": [price * 1.004 for price in closes],
            "low": [price * 0.996 for price in closes],
            "close": closes,
            "volume": [1000.0] * bars
        }
    )


def test_config(short_exposure=0.25):
    return HybridConfig(
        spot_slow_ema_window=10,
        spot_fast_ema_window=5,
        spot_momentum_window=6,
        spot_ema_slope_window=2,
        spot_rebalance_bars=1,
        spot_exposure=0.35,
        short_slow_ema_window=10,
        short_fast_ema_window=5,
        short_momentum_window=6,
        short_ema_slope_window=2,
        short_rebalance_bars=1,
        require_all_symbols_short=False,
        short_exposure=short_exposure,
        atr_window=5,
        short_stop_atr_multiple=30,
        short_cooldown_bars=2
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


class HybridStrategyTests(unittest.TestCase):
    def test_short_exposure_is_hard_capped(self):
        with self.assertRaises(ValueError):
            validate_hybrid_config(
                HybridConfig(short_exposure=1.6)
            )

    def test_future_change_does_not_modify_past_targets(self):
        frames = {
            "BTCUSDT": make_frame(100, 0.003),
            "ETHUSDT": make_frame(100, 0.002)
        }
        config = test_config()
        _, original_targets = generate_hybrid_targets(frames, config)
        changed_frames = {
            symbol: df.copy()
            for symbol, df in frames.items()
        }
        changed_frames["ETHUSDT"].loc[100:, [
            "open", "high", "low", "close"
        ]] *= 4
        _, changed_targets = generate_hybrid_targets(
            changed_frames,
            config
        )

        pd.testing.assert_frame_equal(
            original_targets.iloc[:100],
            changed_targets.iloc[:100]
        )

    def test_rising_market_uses_spot_without_funding(self):
        frames = {
            "BTCUSDT": make_frame(100, 0.003),
            "ETHUSDT": make_frame(100, 0.002)
        }
        funding = empty_funding(frames)
        funding["BTCUSDT"] = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"] * 6,
                "funding_time": frames["BTCUSDT"][
                    "open_time"
                ].iloc[40:46].tolist(),
                "funding_rate": [0.01] * 6,
                "mark_price": frames["BTCUSDT"][
                    "close"
                ].iloc[40:46].tolist()
            }
        )
        result = run_hybrid_backtest(
            frames,
            funding,
            config=test_config(),
            initial_capital=1000,
            spot_fee_rate=0,
            futures_fee_rate=0,
            slippage_rate=0
        )

        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["instrument"] == "spot").all())
        self.assertEqual(result.trades["funding_cash_flow"].sum(), 0)

    def test_falling_market_uses_small_futures_short(self):
        frames = {
            "BTCUSDT": make_frame(100, -0.003),
            "ETHUSDT": make_frame(100, -0.002)
        }
        result = run_hybrid_backtest(
            frames,
            empty_funding(frames),
            config=test_config(short_exposure=0.25),
            initial_capital=1000,
            spot_fee_rate=0,
            futures_fee_rate=0,
            slippage_rate=0
        )

        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["direction"] == "short").all())
        self.assertTrue((result.trades["entry_exposure"] <= 0.25).all())
        self.assertGreater(result.final_value, 1000)

    def test_positive_funding_benefits_short(self):
        frames = {
            "BTCUSDT": make_frame(100, -0.003),
            "ETHUSDT": make_frame(100, -0.002)
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
        free_result = run_hybrid_backtest(
            frames,
            empty_funding(frames),
            config=test_config(),
            spot_fee_rate=0,
            futures_fee_rate=0,
            slippage_rate=0
        )
        funded_result = run_hybrid_backtest(
            frames,
            funding,
            config=test_config(),
            spot_fee_rate=0,
            futures_fee_rate=0,
            slippage_rate=0
        )

        self.assertGreater(
            funded_result.final_value,
            free_result.final_value
        )
        self.assertGreater(
            funded_result.trades["funding_cash_flow"].sum(),
            0
        )
        self.assertLess(funded_result.total_funding_fees, 0)
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
        _, targets = generate_hybrid_targets(frames, config)
        first_signal_position = next(
            i
            for i, row in targets.iterrows()
            if row["target_type"] != "cash"
        )
        result = run_hybrid_backtest(
            frames,
            empty_funding(frames),
            config=config,
            spot_fee_rate=0,
            futures_fee_rate=0,
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
        free_result = run_hybrid_backtest(
            frames,
            empty_funding(frames),
            config=test_config(),
            spot_fee_rate=0,
            futures_fee_rate=0,
            slippage_rate=0
        )
        costly_result = run_hybrid_backtest(
            frames,
            empty_funding(frames),
            config=test_config(),
            spot_fee_rate=0.002,
            futures_fee_rate=0.001,
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
