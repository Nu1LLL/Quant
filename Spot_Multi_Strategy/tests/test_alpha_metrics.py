import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_metrics


ALPHAS_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "alphas"


class ForwardReturnLabelSeparationTests(unittest.TestCase):
    def test_alphas_package_never_imports_alpha_metrics(self):
        for path in ALPHAS_PACKAGE_DIR.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "alpha_metrics",
                source,
                f"{path.name}不应该依赖alpha_metrics（前瞻收益标签模块）"
            )
            self.assertNotIn(
                "forward_return",
                source,
                f"{path.name}不应该出现forward_return（前瞻收益标签），"
                "标签必须与信号计算物理隔离"
            )

    def test_forward_return_only_uses_future_open_prices(self):
        open_price = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
        df = pd.DataFrame({"open": open_price})

        result = alpha_metrics.forward_return(df, horizon=2, delay=1)

        # t=0: entry=open[1]=110, exit=open[3]=133.1
        self.assertAlmostEqual(result.iloc[0], 133.1 / 110.0 - 1)
        # 最后两行没有足够未来数据，必须是NaN，而不是用过去数据伪造
        self.assertTrue(pd.isna(result.iloc[-1]))
        self.assertTrue(pd.isna(result.iloc[-2]))


class AlphaMetricsStatisticsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 400
        self.signal = pd.Series(rng.normal(size=n))
        noise = rng.normal(scale=0.5, size=n)
        # 构造一个signal与forward_return强相关的合成样本
        self.forward_return_series = pd.Series(
            self.signal.values * 1.0 + noise
        )
        self.open_time = pd.date_range(
            "2021-01-01", periods=n, freq="4h", tz="UTC"
        )

    def test_pearson_ic_detects_strong_positive_relationship(self):
        ic, observations = alpha_metrics.pearson_ic(
            self.signal, self.forward_return_series
        )
        self.assertGreater(ic, 0.7)
        self.assertEqual(observations, 400)

    def test_ic_t_stat_grows_with_sample_size(self):
        small_t = alpha_metrics.ic_t_stat(0.3, 30)
        large_t = alpha_metrics.ic_t_stat(0.3, 3000)
        self.assertGreater(large_t, small_t)

    def test_ic_t_stat_handles_degenerate_inputs(self):
        self.assertTrue(pd.isna(alpha_metrics.ic_t_stat(np.nan, 100)))
        self.assertTrue(pd.isna(alpha_metrics.ic_t_stat(0.5, 2)))

    def test_bucket_returns_orders_high_above_low(self):
        buckets = alpha_metrics.bucket_returns(
            self.signal, self.forward_return_series, buckets=3
        )
        self.assertGreater(buckets["high"], buckets["low"])
        self.assertAlmostEqual(
            buckets["spread"], buckets["high"] - buckets["low"]
        )

    def test_turnover_proxy_reflects_signal_stability(self):
        constant_signal = pd.Series([0.5] * 50)
        alternating_signal = pd.Series([0.5, -0.5] * 25)

        self.assertEqual(
            alpha_metrics.turnover_proxy(constant_signal), 0.0
        )
        self.assertGreater(
            alpha_metrics.turnover_proxy(alternating_signal), 0.9
        )

    def test_rolling_ic_positive_ratio_is_high_for_stable_relationship(self):
        ratio, windows_used = alpha_metrics.rolling_ic_positive_ratio(
            self.signal, self.forward_return_series, window=100
        )
        self.assertGreater(ratio, 0.8)
        self.assertGreater(windows_used, 0)

    def test_cost_adjusted_spread_decreases_with_cost_multiplier(self):
        base = alpha_metrics.cost_adjusted_spread(
            spread=0.02,
            turnover=0.1,
            horizon=3,
            fee_rate=0.001,
            slippage_rate=0.0005,
            cost_multiplier=1.0
        )
        stressed = alpha_metrics.cost_adjusted_spread(
            spread=0.02,
            turnover=0.1,
            horizon=3,
            fee_rate=0.001,
            slippage_rate=0.0005,
            cost_multiplier=3.0
        )
        self.assertLess(stressed, base)

    def test_performance_by_year_splits_correctly(self):
        forward_ret = pd.Series(
            [0.01] * 200 + [-0.01] * 200, index=self.signal.index
        )
        two_year_time = pd.date_range(
            "2021-01-01", periods=400, freq="4h", tz="UTC"
        )
        # 把index错开一年，确保有两个日历年
        two_year_time = two_year_time.union(
            pd.date_range("2022-06-01", periods=1, freq="4h", tz="UTC")
        )[:400]

        yearly = alpha_metrics.performance_by_year(
            self.signal, forward_ret, self.open_time
        )
        self.assertIn("year", yearly.columns)
        self.assertGreaterEqual(len(yearly), 1)


if __name__ == "__main__":
    unittest.main()
