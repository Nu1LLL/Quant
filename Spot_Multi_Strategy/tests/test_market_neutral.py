import unittest

import numpy as np
import pandas as pd

import alphas
import market_neutral as mn


def _make_ohlcv(rows, seed, start_price=100.0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=0.01, size=rows)
    close = start_price * np.cumprod(1 + returns)

    open_ = np.empty(rows)
    open_[0] = start_price
    open_[1:] = close[:-1]

    open_time = pd.date_range(start, periods=rows, freq="4h", tz="UTC")

    return pd.DataFrame({
        "open_time": open_time,
        "open": open_,
        "high": np.maximum(open_, close) * 1.002,
        "low": np.minimum(open_, close) * 0.998,
        "close": close,
        "volume": rng.uniform(100, 1000, rows)
    })


class CrossSectionalConstructionTests(unittest.TestCase):
    def setUp(self):
        self.frames = {
            "AAAUSDT": _make_ohlcv(1000, seed=1, start_price=100.0),
            "BBBUSDT": _make_ohlcv(1000, seed=2, start_price=50.0),
            "CCCUSDT": _make_ohlcv(1000, seed=3, start_price=10.0),
        }
        self.alpha_sets = {
            symbol: alphas.build_single_asset_alphas(df)
            for symbol, df in self.frames.items()
        }

    def test_cross_sectional_demean_sums_to_zero_each_row(self):
        signal = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A02_ema_distance_50"
        )
        row_sums = signal.dropna().sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 0.0, atol=1e-9))

    def test_gross_cap_never_exceeds_cap_and_never_scales_up(self):
        signal = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A02_ema_distance_50"
        )
        capped = mn.scale_to_gross_cap(signal, gross_cap=1.0)
        gross = capped.abs().sum(axis=1)
        self.assertTrue((gross <= 1.0 + 1e-9).all())

        # 缩放只能缩小，不能放大：任何一行的绝对值都不应该比裁剪前更大
        # （NaN比较本身恒为False，用fillna(True)避免NaN单元格误报）
        raw_clipped = signal.clip(-1, 1)
        comparison = (capped.abs() <= raw_clipped.abs() + 1e-9).where(
            raw_clipped.notna(), True
        )
        self.assertTrue(comparison.all().all())

    def test_no_trade_band_reduces_turnover_events(self):
        signal = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A02_ema_distance_50"
        )
        capped = mn.scale_to_gross_cap(signal, gross_cap=1.0)

        _, tight_turnover = mn.apply_no_trade_band(capped, no_trade_band=0.0)
        _, loose_turnover = mn.apply_no_trade_band(capped, no_trade_band=0.3)

        self.assertGreater(
            (tight_turnover > 0).sum().sum(),
            (loose_turnover > 0).sum().sum()
        )

    def test_late_starting_asset_excluded_before_its_own_history(self):
        late_frames = dict(self.frames)
        late_frames["DDDUSDT"] = _make_ohlcv(
            1000, seed=4, start_price=1.0
        ).iloc[200:].reset_index(drop=True)
        late_alpha_sets = dict(self.alpha_sets)
        late_alpha_sets["DDDUSDT"] = alphas.build_single_asset_alphas(
            late_frames["DDDUSDT"]
        )

        signal = mn.build_cross_sectional_signal(
            late_frames, late_alpha_sets, "A02_ema_distance_50"
        )
        early_rows = signal.iloc[:50]
        self.assertTrue(early_rows["DDDUSDT"].isna().all())
        # 其它资产不应该因为DDD还没数据就被污染成全NaN
        self.assertFalse(early_rows["AAAUSDT"].isna().all())


class RankConstructionTests(unittest.TestCase):
    def setUp(self):
        self.frames = {
            "AAAUSDT": _make_ohlcv(1000, seed=21, start_price=100.0),
            "BBBUSDT": _make_ohlcv(1000, seed=22, start_price=50.0),
            "CCCUSDT": _make_ohlcv(1000, seed=23, start_price=10.0),
            "DDDUSDT": _make_ohlcv(1000, seed=24, start_price=5.0),
        }
        self.alpha_sets = {
            symbol: alphas.build_single_asset_alphas(df)
            for symbol, df in self.frames.items()
        }

    def test_rank_demean_is_bounded_and_sums_near_zero(self):
        signal = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A02_ema_distance_50",
            method="rank"
        )
        valid = signal.dropna()
        self.assertTrue((valid.abs() <= 1.0 + 1e-9).all().all())
        # 4个资产的完整排名去中心化后每行和应该恒为0
        self.assertTrue(np.allclose(valid.sum(axis=1), 0.0, atol=1e-9))

    def test_rank_preserves_ordering_ignores_magnitude(self):
        # 构造一个已知顺序的横截面分数，验证排名版本只保留顺序信息
        master = pd.DataFrame({
            "AAAUSDT": [100.0], "BBBUSDT": [1.0],
            "CCCUSDT": [0.5], "DDDUSDT": [-50.0]
        })
        ranked = mn.cross_sectional_rank_demean(master)
        row = ranked.iloc[0]
        self.assertGreater(row["AAAUSDT"], row["BBBUSDT"])
        self.assertGreater(row["BBBUSDT"], row["CCCUSDT"])
        self.assertGreater(row["CCCUSDT"], row["DDDUSDT"])
        # AAA领先BBB100倍和领先2倍，排名结果应该完全一样（对幅度不敏感）
        alt_master = pd.DataFrame({
            "AAAUSDT": [2.0], "BBBUSDT": [1.0],
            "CCCUSDT": [0.5], "DDDUSDT": [-50.0]
        })
        alt_ranked = mn.cross_sectional_rank_demean(alt_master)
        pd.testing.assert_series_equal(
            ranked.iloc[0], alt_ranked.iloc[0], check_names=False
        )

    def test_rank_method_rejects_unknown_method_name(self):
        with self.assertRaises(ValueError):
            mn.build_cross_sectional_signal(
                self.frames, self.alpha_sets, "A02_ema_distance_50",
                method="not_a_real_method"
            )


class LiteratureMomentumAlphaTests(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(1000, seed=30)
        self.alphas = alphas.build_single_asset_alphas(self.df)

    def test_skip_momentum_alphas_present_and_bounded(self):
        skip_names = [
            name for name in self.alphas if name.startswith("A24_")
        ]
        self.assertGreater(len(skip_names), 0)
        for name in skip_names:
            values = self.alphas[name].normalized_signal.dropna()
            self.assertTrue((values.abs() <= 1.0 + 1e-9).all())

    def test_sign_tsmom_alphas_are_exactly_signed(self):
        sign_names = [
            name for name in self.alphas if name.startswith("A25_")
        ]
        self.assertGreater(len(sign_names), 0)
        for name in sign_names:
            values = self.alphas[name].normalized_signal.dropna()
            self.assertTrue(values.isin([-1.0, 0.0, 1.0]).all())

    def test_skip_momentum_formula_excludes_the_skip_window(self):
        # 直接按公式验证：第i行的原始值应该等于close[i-skip]/close[i-skip-horizon]-1，
        # 完全不涉及close[i-skip+1..i]这skip根最近的K线。
        import alphas.momentum as momentum_module

        horizon, skip = 72, 6
        signals = momentum_module.build_skip_period_momentum(
            self.df, horizons=(horizon,), skip=skip
        )
        raw = signals[f"A24_skip_momentum_{horizon}_{skip}"].raw_signal

        close = self.df["close"]
        expected = close.shift(skip) / close.shift(skip + horizon) - 1

        pd.testing.assert_series_equal(raw, expected, check_names=False)

        # 再直接证明：篡改close[i-skip+1..i]这一段不会改变第i行的值——
        # 挑一个中间行i，只改它自己往前skip根以内的价格。
        i = 500
        mutated_df = self.df.copy()
        mutated_df.loc[i - skip + 1:i, "close"] *= 3.0
        mutated_signals = momentum_module.build_skip_period_momentum(
            mutated_df, horizons=(horizon,), skip=skip
        )
        mutated_raw = mutated_signals[
            f"A24_skip_momentum_{horizon}_{skip}"
        ].raw_signal
        self.assertAlmostEqual(raw.iloc[i], mutated_raw.iloc[i])


class RebalanceScheduleTests(unittest.TestCase):
    def test_bars_between_rebalance_points_hold_the_last_scheduled_value(self):
        matrix = pd.DataFrame({
            "AAAUSDT": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        })
        resampled = mn.resample_to_rebalance_schedule(matrix, 3)
        # 第0,3,6行是调仓点，中间沿用上一次调仓点的值
        self.assertEqual(
            resampled["AAAUSDT"].tolist(),
            [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 7.0]
        )

    def test_every_bar_default_is_a_no_op(self):
        matrix = pd.DataFrame({"AAAUSDT": [1.0, 2.0, 3.0]})
        resampled = mn.resample_to_rebalance_schedule(matrix, 1)
        pd.testing.assert_frame_equal(resampled, matrix)

    def test_resample_is_causal(self):
        rng = np.random.default_rng(40)
        matrix = pd.DataFrame({"AAAUSDT": rng.normal(size=200)})
        full = mn.resample_to_rebalance_schedule(matrix, 6)
        truncated = mn.resample_to_rebalance_schedule(matrix.iloc[:-50], 6)
        pd.testing.assert_frame_equal(full.iloc[:-50], truncated)


class NoLookaheadTests(unittest.TestCase):
    def setUp(self):
        self.frames = {
            "AAAUSDT": _make_ohlcv(800, seed=5, start_price=100.0),
            "BBBUSDT": _make_ohlcv(800, seed=6, start_price=50.0),
            "CCCUSDT": _make_ohlcv(800, seed=7, start_price=10.0),
        }
        self.alpha_sets = {
            symbol: alphas.build_single_asset_alphas(df)
            for symbol, df in self.frames.items()
        }

    def test_signal_truncation_by_calendar_time_is_causal(self):
        full = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A08_trend_quality"
        )
        cutoff = self.frames["AAAUSDT"]["open_time"].iloc[-100]
        truncated_frames = {
            symbol: frame[frame["open_time"] < cutoff].reset_index(drop=True)
            for symbol, frame in self.frames.items()
        }
        truncated_alpha_sets = {
            symbol: alphas.build_single_asset_alphas(df)
            for symbol, df in truncated_frames.items()
        }
        truncated = mn.build_cross_sectional_signal(
            truncated_frames, truncated_alpha_sets, "A08_trend_quality"
        )

        overlap = truncated.index
        pd.testing.assert_frame_equal(
            full.loc[overlap], truncated, check_dtype=False
        )

    def test_apply_no_trade_band_is_causal(self):
        signal = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A08_trend_quality"
        )
        capped = mn.scale_to_gross_cap(signal)

        full_pos, _ = mn.apply_no_trade_band(capped, no_trade_band=0.05)

        cutoff_position = len(capped) - 100
        truncated_capped = capped.iloc[:cutoff_position]
        truncated_pos, _ = mn.apply_no_trade_band(
            truncated_capped, no_trade_band=0.05
        )

        pd.testing.assert_frame_equal(
            full_pos.iloc[:cutoff_position], truncated_pos, check_dtype=False
        )


class FundingCostSignTests(unittest.TestCase):
    def test_long_pays_when_funding_positive_short_receives(self):
        rows = 200
        frame = _make_ohlcv(rows, seed=8)
        funding_time = pd.date_range(
            "2020-01-01", periods=rows // 2, freq="8h", tz="UTC"
        )
        funding_df = pd.DataFrame({
            "funding_time": funding_time,
            "funding_rate": np.full(len(funding_time), 0.001)
        })

        frames = {"AAAUSDT": frame}
        funding_frames = {"AAAUSDT": funding_df}

        long_weights = pd.DataFrame(
            {"AAAUSDT": np.full(rows, 0.5)},
            index=pd.to_datetime(frame["open_time"], utc=True)
        )
        short_weights = pd.DataFrame(
            {"AAAUSDT": np.full(rows, -0.5)},
            index=pd.to_datetime(frame["open_time"], utc=True)
        )

        long_result = mn.simulate_market_neutral_portfolio(
            frames, funding_frames, long_weights, no_trade_band=0.0
        )
        short_result = mn.simulate_market_neutral_portfolio(
            frames, funding_frames, short_weights, no_trade_band=0.0
        )

        long_funding = long_result["AAAUSDT"]["funding_pnl"].iloc[1:-3].sum()
        short_funding = short_result["AAAUSDT"]["funding_pnl"].iloc[1:-3].sum()

        self.assertLess(long_funding, 0)
        self.assertGreater(short_funding, 0)
        self.assertAlmostEqual(long_funding, -short_funding, places=6)


class GateSmokeTests(unittest.TestCase):
    def test_evaluate_cross_sectional_alpha_runs_and_reports_checks(self):
        frames = {
            "AAAUSDT": _make_ohlcv(3500, seed=9, start_price=100.0),
            "BBBUSDT": _make_ohlcv(3500, seed=10, start_price=50.0),
            "CCCUSDT": _make_ohlcv(3500, seed=11, start_price=10.0),
        }
        alpha_sets = {
            symbol: alphas.build_single_asset_alphas(df)
            for symbol, df in frames.items()
        }
        result = mn.evaluate_cross_sectional_alpha(
            "A02_ema_distance_50", frames, alpha_sets,
            min_total_observations=100, min_fold_observations=10
        )
        self.assertIsInstance(result.passed, (bool, np.bool_))
        self.assertEqual(len(result.checks), 7)
        self.assertEqual(len(result.fold_table), 6)


if __name__ == "__main__":
    unittest.main()
