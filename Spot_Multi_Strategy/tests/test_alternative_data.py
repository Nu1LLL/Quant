import unittest

import numpy as np
import pandas as pd

import alphas.alternative_data as alt


def _make_ohlcv(rows, seed, start_price=100.0, start="2021-01-01"):
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


def _make_funding(rows_4h, seed, start="2021-01-01"):
    # 资金费率每8小时结算一次，比4小时K线稀疏一半
    rng = np.random.default_rng(seed)
    funding_time = pd.date_range(
        start, periods=rows_4h // 2, freq="8h", tz="UTC"
    )
    funding_rate = rng.normal(scale=0.0003, size=len(funding_time))
    return pd.DataFrame({
        "symbol": "TESTUSDT",
        "funding_time": funding_time,
        "funding_rate": funding_rate,
        "mark_price": 0.0
    })


class FundingAlignmentTests(unittest.TestCase):
    def test_alignment_only_uses_already_settled_funding(self):
        df = _make_ohlcv(20, seed=1)
        funding = pd.DataFrame({
            "funding_time": pd.to_datetime([
                "2021-01-01 00:00:00", "2021-01-01 08:00:00"
            ], utc=True),
            "funding_rate": [0.001, -0.002]
        })

        aligned = alt.align_funding_rate_to_bars(df, funding)

        # 2021-01-01 00:00的K线开盘时刚好等于第一次结算时刻，应该拿到0.001
        self.assertAlmostEqual(aligned.iloc[0], 0.001)
        # 04:00的K线还没到第二次结算(08:00)，应该沿用上一次的0.001
        self.assertAlmostEqual(aligned.iloc[1], 0.001)
        # 08:00及以后应该用新结算的-0.002
        self.assertAlmostEqual(aligned.iloc[2], -0.002)

    def test_bars_before_any_settlement_are_nan(self):
        df = _make_ohlcv(5, seed=2, start="2020-01-01")
        funding = pd.DataFrame({
            "funding_time": pd.to_datetime(["2020-01-05"], utc=True),
            "funding_rate": [0.001]
        })
        aligned = alt.align_funding_rate_to_bars(df, funding)
        self.assertTrue(aligned.isna().all())


class FundingRateAlphaTests(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(600, seed=3)
        self.funding = _make_funding(600, seed=4)

    def test_alphas_are_bounded_and_share_index(self):
        result = alt.build_alphas(self.df, self.funding)
        self.assertEqual(len(result), 2)
        for name, signal in result.items():
            self.assertTrue(
                signal.raw_signal.index.equals(signal.normalized_signal.index)
            )
            values = signal.normalized_signal.dropna()
            self.assertTrue((values.abs() <= 1.0 + 1e-9).all())

    def test_input_frames_are_not_mutated(self):
        df_before = self.df.copy(deep=True)
        funding_before = self.funding.copy(deep=True)
        alt.build_alphas(self.df, self.funding)
        pd.testing.assert_frame_equal(self.df, df_before)
        pd.testing.assert_frame_equal(self.funding, funding_before)

    def test_no_lookahead_truncating_both_frames_by_calendar_time(self):
        full = alt.build_alphas(self.df, self.funding)

        cutoff = self.df["open_time"].iloc[-100]
        truncated_df = self.df[
            self.df["open_time"] < cutoff
        ].reset_index(drop=True)
        truncated_funding = self.funding[
            self.funding["funding_time"] < cutoff
        ].reset_index(drop=True)

        truncated = alt.build_alphas(truncated_df, truncated_funding)

        for name, signal in truncated.items():
            full_prefix = (
                full[name].normalized_signal
                .iloc[:len(truncated_df)]
                .reset_index(drop=True)
            )
            pd.testing.assert_series_equal(
                full_prefix,
                signal.normalized_signal.reset_index(drop=True),
                check_names=False
            )

    def test_crowding_and_trend_have_opposite_sign_conventions(self):
        # 前半段资金费率在0附近小幅波动，建立"正常水平"的滚动基线；
        # 后半段突然跳升到持续偏高——A21衡量的是"相对近期分布的异常
        # 偏离"，应该在跳升后给出负（反向/拥挤）分数；A22衡量的是
        # "近期滚动均值方向"，应该给出正（顺势）分数。两个假设方向
        # 相反是设计意图，不是bug。
        rows = 800
        df = _make_ohlcv(rows, seed=5)
        settlement_count = rows // 2
        funding_time = pd.date_range(
            "2021-01-01", periods=settlement_count, freq="8h", tz="UTC"
        )
        rng = np.random.default_rng(6)
        baseline = rng.normal(scale=0.00005, size=settlement_count)
        jump_point = int(settlement_count * 0.8)
        baseline[jump_point:] += 0.01
        funding = pd.DataFrame({
            "funding_time": funding_time,
            "funding_rate": baseline
        })

        result = alt.build_alphas(df, funding)
        crowding = result["A21_funding_rate_crowding_90"].normalized_signal
        trend = result["A22_funding_rate_trend_21"].normalized_signal

        self.assertLess(crowding.dropna().iloc[-1], 0)
        self.assertGreater(trend.dropna().iloc[-1], 0)


if __name__ == "__main__":
    unittest.main()
