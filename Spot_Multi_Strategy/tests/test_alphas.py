import unittest

import numpy as np
import pandas as pd

import alphas


def _make_ohlcv(rows=400, seed=0, start_price=100.0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=0.01, size=rows)
    close = start_price * np.cumprod(1 + returns)

    open_ = np.empty(rows)
    open_[0] = start_price
    open_[1:] = close[:-1]

    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, rows))
    volume = rng.uniform(100, 1000, rows)

    open_time = pd.date_range(
        "2021-01-01",
        periods=rows,
        freq="4h",
        tz="UTC"
    )

    return pd.DataFrame({
        "open_time": open_time,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })


class SingleAssetAlphaTests(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(rows=400, seed=1)
        self.alphas = alphas.build_single_asset_alphas(self.df)

    def test_at_least_one_alpha_per_documented_family(self):
        prefixes = {
            "A01", "A02", "A03", "A04", "A05", "A06", "A07",
            "A08", "A09", "A10", "A11", "A12", "A13", "A14",
            "A15", "A16", "A17"
        }
        found_prefixes = {name.split("_")[0] for name in self.alphas}
        missing = prefixes - found_prefixes
        self.assertEqual(missing, set())

    def test_normalized_signal_is_bounded(self):
        for name, signal in self.alphas.items():
            values = signal.normalized_signal.dropna()
            self.assertTrue(
                (values.abs() <= 1.0 + 1e-9).all(),
                f"{name}未落在[-1,1]范围内"
            )

    def test_raw_and_normalized_share_index(self):
        for name, signal in self.alphas.items():
            self.assertTrue(
                signal.raw_signal.index.equals(
                    signal.normalized_signal.index
                ),
                f"{name}的raw_signal与normalized_signal索引不一致"
            )

    def test_input_dataframe_is_not_mutated(self):
        before = self.df.copy(deep=True)
        alphas.build_single_asset_alphas(self.df)
        pd.testing.assert_frame_equal(self.df, before)

    def test_no_lookahead_truncating_tail_does_not_change_past_values(self):
        truncated_df = self.df.iloc[:-100].reset_index(drop=True)
        truncated_alphas = alphas.build_single_asset_alphas(truncated_df)

        for name, signal in truncated_alphas.items():
            full_prefix = (
                self.alphas[name]
                .normalized_signal
                .iloc[:len(truncated_df)]
                .reset_index(drop=True)
            )
            truncated_values = signal.normalized_signal.reset_index(
                drop=True
            )

            pd.testing.assert_series_equal(
                full_prefix,
                truncated_values,
                check_names=False,
                obj=f"{name}: 截断尾部数据后过去的值发生了变化"
            )

    def test_ensemble_alphas_only_use_declared_components(self):
        momentum_ensemble = self.alphas["A16_momentum_ensemble"]
        component_names = momentum_ensemble.metadata["component_alphas"]

        self.assertTrue(
            all(name.startswith("A01_") for name in component_names)
        )

        expected = pd.concat(
            [
                self.alphas[name].normalized_signal
                for name in component_names
            ],
            axis=1
        ).mean(axis=1)

        pd.testing.assert_series_equal(
            momentum_ensemble.normalized_signal,
            expected,
            check_names=False
        )


class CrossAssetAlphaTests(unittest.TestCase):
    def setUp(self):
        self.btc_df = _make_ohlcv(rows=300, seed=2, start_price=30000.0)
        self.eth_df = _make_ohlcv(rows=300, seed=3, start_price=2000.0)
        # 强制两个资产共用同一组时间戳，模拟真实数据对齐后的样子
        self.eth_df["open_time"] = self.btc_df["open_time"]

    def test_rejects_misaligned_timestamps(self):
        misaligned_eth = self.eth_df.copy()
        misaligned_eth.loc[5, "open_time"] = (
            misaligned_eth.loc[5, "open_time"] + pd.Timedelta(hours=4)
        )

        with self.assertRaises(ValueError):
            alphas.build_cross_asset_alphas(self.btc_df, misaligned_eth)

    def test_rejects_mismatched_length(self):
        shorter_eth = self.eth_df.iloc[:-10]

        with self.assertRaises(ValueError):
            alphas.build_cross_asset_alphas(self.btc_df, shorter_eth)

    def test_btc_lead_alpha_only_uses_past_btc_returns(self):
        cross_alphas = alphas.build_cross_asset_alphas(
            self.btc_df,
            self.eth_df
        )
        lag_1 = cross_alphas["A18_btc_lead_eth_lag_1"]

        btc_close = self.btc_df["close"]
        expected_raw = (btc_close / btc_close.shift(1) - 1).reset_index(
            drop=True
        )

        pd.testing.assert_series_equal(
            lag_1.raw_signal.reset_index(drop=True),
            expected_raw,
            check_names=False
        )

        # 篡改ETH自己的收盘价（未来ETH信息）不应该影响这个alpha，
        # 因为它只依赖BTC历史收益
        tampered_eth = self.eth_df.copy()
        tampered_eth["close"] = tampered_eth["close"] * 1.5
        tampered_alphas = alphas.build_cross_asset_alphas(
            self.btc_df,
            tampered_eth
        )

        pd.testing.assert_series_equal(
            lag_1.raw_signal,
            tampered_alphas["A18_btc_lead_eth_lag_1"].raw_signal,
            check_names=False
        )

    def test_full_alpha_library_attaches_cross_asset_only_when_requested(self):
        eth_only = alphas.build_single_asset_alphas(self.eth_df)
        combined = alphas.build_alpha_library(
            self.eth_df,
            btc_df=self.btc_df,
            eth_df=self.eth_df
        )

        self.assertTrue(
            any(name.startswith("A18_") for name in combined)
        )
        self.assertFalse(
            any(name.startswith("A18_") for name in eth_only)
        )


class MultiAssetCrossAssetAlphaTests(unittest.TestCase):
    """A20市场广度、A23跨资产相对强弱：验证不同长度历史（比如某个
    资产比其他资产晚上线）不会互相污染，也不会引入未来数据泄漏。
    """

    def setUp(self):
        self.frames = {
            "AAAUSDT": _make_ohlcv(rows=500, seed=11, start_price=100.0),
            "BBBUSDT": _make_ohlcv(rows=500, seed=12, start_price=50.0),
            "CCCUSDT": _make_ohlcv(rows=500, seed=13, start_price=10.0),
        }
        # 让第四个资产比其他资产晚150根K线才有数据，模拟SOL晚于
        # BTC/ETH/BNB在Binance上市的情况
        late_asset = _make_ohlcv(rows=500, seed=14, start_price=1.0)
        self.frames["DDDUSDT"] = late_asset.iloc[150:].reset_index(drop=True)

    def test_breadth_and_relative_strength_handle_unequal_length_history(self):
        breadth = alphas.cross_asset.build_market_breadth(self.frames)
        relative_strength = (
            alphas.cross_asset.build_cross_sectional_relative_strength(
                self.frames
            )
        )

        for symbol, frame in self.frames.items():
            self.assertEqual(
                len(breadth[symbol]["A20_market_breadth_24"].normalized_signal),
                len(frame)
            )
            self.assertEqual(
                len(
                    relative_strength[symbol][
                        "A23_cross_sectional_relative_strength_24"
                    ].normalized_signal
                ),
                len(frame)
            )

    def test_late_starting_asset_does_not_produce_values_before_its_own_history(self):
        relative_strength = (
            alphas.cross_asset.build_cross_sectional_relative_strength(
                self.frames
            )
        )
        signal = relative_strength["DDDUSDT"][
            "A23_cross_sectional_relative_strength_24"
        ]
        # DDD自己的动量需要24根K线回看，加上波动率48根K线窗口，
        # 之前应该都是NaN，不应该因为其他资产更早有数据就被填出数值
        self.assertTrue(signal.normalized_signal.iloc[:47].isna().all())

    def test_no_lookahead_truncating_all_frames_by_calendar_time(self):
        full_breadth = alphas.cross_asset.build_market_breadth(self.frames)
        full_rs = alphas.cross_asset.build_cross_sectional_relative_strength(
            self.frames
        )

        cutoff = self.frames["AAAUSDT"]["open_time"].iloc[-100]
        truncated_frames = {
            symbol: frame[frame["open_time"] < cutoff].reset_index(drop=True)
            for symbol, frame in self.frames.items()
        }
        truncated_breadth = alphas.cross_asset.build_market_breadth(
            truncated_frames
        )
        truncated_rs = (
            alphas.cross_asset.build_cross_sectional_relative_strength(
                truncated_frames
            )
        )

        for symbol, frame in truncated_frames.items():
            for full_result, truncated_result, alpha_name in [
                (
                    full_breadth, truncated_breadth,
                    "A20_market_breadth_24"
                ),
                (
                    full_rs, truncated_rs,
                    "A23_cross_sectional_relative_strength_24"
                ),
            ]:
                full_prefix = (
                    full_result[symbol][alpha_name].normalized_signal
                    .iloc[:len(frame)]
                    .reset_index(drop=True)
                )
                truncated_values = (
                    truncated_result[symbol][alpha_name].normalized_signal
                    .reset_index(drop=True)
                )
                pd.testing.assert_series_equal(
                    full_prefix, truncated_values, check_names=False
                )

    def test_breadth_reflects_unanimous_direction(self):
        rows = 200
        trending_up = {
            symbol: _make_ohlcv(rows=rows, seed=20 + i, start_price=100.0)
            for i, symbol in enumerate(["X1USDT", "X2USDT", "X3USDT"])
        }
        # 强制三个资产在最后一段时间同步大涨
        for frame in trending_up.values():
            frame.loc[rows - 30:, "close"] = (
                frame.loc[rows - 31, "close"]
                * np.cumprod(np.full(30, 1.05))
            )
            frame.loc[rows - 30:, "open"] = frame.loc[rows - 30:, "close"]

        breadth = alphas.cross_asset.build_market_breadth(
            trending_up, horizon=10
        )
        last_breadth = breadth["X1USDT"][
            "A20_market_breadth_10"
        ].raw_signal.iloc[-1]
        self.assertAlmostEqual(last_breadth, 1.0)


if __name__ == "__main__":
    unittest.main()
