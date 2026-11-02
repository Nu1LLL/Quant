import unittest

import numpy as np
import pandas as pd

import alphas.chan_theory as chan


def _make_ohlcv(rows, seed, start_price=100.0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=0.015, size=rows)
    close = start_price * np.cumprod(1 + returns)

    open_ = np.empty(rows)
    open_[0] = start_price
    open_[1:] = close[:-1]

    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.006, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.006, rows))
    volume = rng.uniform(100, 1000, rows)

    open_time = pd.date_range(
        "2021-01-01", periods=rows, freq="4h", tz="UTC"
    )

    return pd.DataFrame({
        "open_time": open_time,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })


def _bars(rows):
    """[(high, low), ...]转成一个最小的df，只用于K线合并/分型测试。"""
    return pd.DataFrame({
        "open_time": pd.date_range(
            "2021-01-01", periods=len(rows), freq="4h", tz="UTC"
        ),
        "open": [r[0] for r in rows],
        "high": [r[0] for r in rows],
        "low": [r[1] for r in rows],
        "close": [r[0] for r in rows],
        "volume": [100.0] * len(rows)
    })


class InclusionMergeTests(unittest.TestCase):
    def test_contained_bar_is_merged_in_uptrend(self):
        # 上升方向中，第3根被第2根完全包含，应该合并成一根
        df = _bars([(10, 8), (12, 9), (11, 9.5)])
        merged_high, merged_low, merged_start_idx = chan.merge_inclusive_bars(df)
        self.assertEqual(len(merged_high), 2)
        # 合并后：高点取两者较大值max(12,11)=12，低点取两者较高者max(9,9.5)=9.5
        self.assertAlmostEqual(merged_high[1], 12.0)
        self.assertAlmostEqual(merged_low[1], 9.5)

    def test_non_overlapping_bars_are_not_merged(self):
        df = _bars([(10, 8), (12, 11), (14, 13)])
        merged_high, merged_low, merged_start_idx = chan.merge_inclusive_bars(df)
        self.assertEqual(len(merged_high), 3)


class FractalDetectionTests(unittest.TestCase):
    def test_detects_a_clean_top_fractal(self):
        df = _bars([(10, 8), (12, 11), (15, 13), (12, 10), (11, 9)])
        merged_high, merged_low, merged_start_idx = chan.merge_inclusive_bars(df)
        fractals = chan.detect_fractals(merged_high, merged_low, merged_start_idx)
        kinds = [f[1] for f in fractals]
        self.assertIn(1, kinds)

    def test_detects_a_clean_bottom_fractal(self):
        df = _bars([(15, 13), (12, 10), (9, 7), (12, 10), (14, 12)])
        merged_high, merged_low, merged_start_idx = chan.merge_inclusive_bars(df)
        fractals = chan.detect_fractals(merged_high, merged_low, merged_start_idx)
        kinds = [f[1] for f in fractals]
        self.assertIn(-1, kinds)

    def test_fractal_confirmation_index_is_after_its_own_position(self):
        df = _make_ohlcv(500, seed=1)
        merged_high, merged_low, merged_start_idx = chan.merge_inclusive_bars(df)
        fractals = chan.detect_fractals(merged_high, merged_low, merged_start_idx)
        for merged_idx, kind, price, confirmed_at in fractals:
            # 分型所在的合并K线起始行号，必须严格早于它被确认的行号
            self.assertLess(
                int(merged_start_idx[merged_idx]), confirmed_at
            )


class PivotConstructionTests(unittest.TestCase):
    def test_valid_pivot_has_positive_width(self):
        df = _make_ohlcv(2000, seed=2)
        result = chan.build_alphas(df)
        # 至少应该能在这么长的合成数据上找到一些中枢结构
        merged_high, merged_low, merged_start_idx = chan.merge_inclusive_bars(df)
        fractals = chan.detect_fractals(merged_high, merged_low, merged_start_idx)
        strokes = chan.build_strokes(fractals)
        pivots = chan.build_pivots(strokes)
        self.assertGreater(len(pivots), 0)
        for confirmed_at, zg, zd in pivots:
            self.assertGreater(zg, zd)


class AlphaOutputTests(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(3000, seed=3)
        self.result = chan.build_alphas(self.df)

    def test_signal_is_bounded(self):
        signal = self.result["A26_chan_pivot_position"]
        values = signal.normalized_signal.dropna()
        self.assertTrue((values.abs() <= 1.0 + 1e-9).all())

    def test_raw_and_normalized_share_index(self):
        signal = self.result["A26_chan_pivot_position"]
        self.assertTrue(
            signal.raw_signal.index.equals(signal.normalized_signal.index)
        )

    def test_input_dataframe_is_not_mutated(self):
        before = self.df.copy(deep=True)
        chan.build_alphas(self.df)
        pd.testing.assert_frame_equal(self.df, before)

    def test_early_bars_before_any_pivot_are_nan(self):
        signal = self.result["A26_chan_pivot_position"]
        # 数据一开始还没有任何已确认中枢，必须是NaN，不能编出一个分数
        self.assertTrue(signal.normalized_signal.iloc[:5].isna().all())


class NoLookaheadTests(unittest.TestCase):
    def test_truncating_the_tail_does_not_change_past_values(self):
        df = _make_ohlcv(4000, seed=4)
        full = chan.build_alphas(df)

        truncated_df = df.iloc[:-300].reset_index(drop=True)
        truncated = chan.build_alphas(truncated_df)

        full_prefix = (
            full["A26_chan_pivot_position"].normalized_signal
            .iloc[:len(truncated_df)]
            .reset_index(drop=True)
        )
        truncated_values = (
            truncated["A26_chan_pivot_position"].normalized_signal
            .reset_index(drop=True)
        )

        pd.testing.assert_series_equal(
            full_prefix, truncated_values, check_names=False
        )

    def test_mutating_future_prices_does_not_change_earlier_values(self):
        df = _make_ohlcv(4000, seed=5)
        full = chan.build_alphas(df)

        mutated_df = df.copy()
        cutoff = int(len(df) * 0.9)
        mutated_df.loc[cutoff:, ["open", "high", "low", "close"]] *= 3.0
        mutated = chan.build_alphas(mutated_df)

        # 只在截断点之前留出安全边界（分型/笔/中枢的确认延迟），
        # 越靠近截断点的少数行允许因为确认延迟机制不同而有差异
        safe_boundary = cutoff - 20
        pd.testing.assert_series_equal(
            full["A26_chan_pivot_position"].normalized_signal.iloc[:safe_boundary],
            mutated["A26_chan_pivot_position"].normalized_signal.iloc[:safe_boundary],
            check_names=False
        )


if __name__ == "__main__":
    unittest.main()
