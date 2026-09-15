import unittest

import numpy as np
import pandas as pd

import golden_cross_oos as gc


def _prices(dates, returns, start=100.0):
    return pd.Series(start * np.cumprod(1.0 + np.asarray(returns)), index=dates)


class ValidatePricesTests(unittest.TestCase):
    def _dates(self, n):
        return pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    def test_rejects_duplicate_dates(self):
        dates = self._dates(5)
        prices = pd.Series([1.0] * 5, index=list(dates[:-1]) + [dates[-2]])
        with self.assertRaises(ValueError):
            gc.validate_prices(prices)

    def test_rejects_non_positive_prices(self):
        dates = self._dates(5)
        prices = pd.Series([1.0, 1.0, -1.0, 1.0, 1.0], index=dates)
        with self.assertRaises(ValueError):
            gc.validate_prices(prices)

    def test_rejects_abnormally_long_gap(self):
        dates = list(self._dates(3)) + [pd.Timestamp("2021-06-01", tz="UTC")]
        prices = pd.Series([1.0] * 4, index=dates)
        with self.assertRaises(ValueError):
            gc.validate_prices(prices)


class BuildTargetPositionsTests(unittest.TestCase):
    def test_no_position_before_slow_window_is_full(self):
        dates = pd.bdate_range("2020-01-01", periods=250, tz="UTC")
        rng = np.random.default_rng(239)
        prices = _prices(dates, rng.normal(0.0003, 0.01, 250))
        target = gc.build_target_positions(prices, leverage=1.0)
        self.assertTrue((target.iloc[:200] == 0.0).all())

    def test_uses_prior_day_ma_relationship_not_same_day(self):
        # 构造一段明确的上升趋势，快线最终应该上穿慢线；用手动
        # 复现的shift(1)版本核对，而不是重新实现同样的公式
        dates = pd.bdate_range("2020-01-01", periods=260, tz="UTC")
        rng = np.random.default_rng(241)
        prices = _prices(dates, rng.normal(0.002, 0.005, 260))
        target = gc.build_target_positions(prices, leverage=1.0)
        fast_ma = prices.rolling(50, min_periods=50).mean()
        slow_ma = prices.rolling(200, min_periods=200).mean()
        expected = (fast_ma > slow_ma).shift(1).fillna(False).astype(float)
        pd.testing.assert_series_equal(target, expected, check_names=False)

    def test_leverage_scales_target_when_golden_cross_active(self):
        dates = pd.bdate_range("2020-01-01", periods=260, tz="UTC")
        rng = np.random.default_rng(243)
        prices = _prices(dates, rng.normal(0.002, 0.004, 260))
        target_1x = gc.build_target_positions(prices, leverage=1.0)
        target_2x = gc.build_target_positions(prices, leverage=2.0)
        active_days = target_1x[target_1x > 0].index
        if len(active_days):
            self.assertTrue((target_2x.loc[active_days] == 2.0).all())


class RunScenarioTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2020-01-01", periods=400, tz="UTC")
        rng = np.random.default_rng(251)
        self.returns_input = rng.normal(0.0004, 0.011, len(self.dates))
        self.prices = _prices(self.dates, self.returns_input)

    def test_no_leverage_financing_at_1x(self):
        _, detail = gc.run_scenario(self.prices, leverage=1.0)
        self.assertTrue((detail["leverage_financing"] == 0.0).all())

    def test_leverage_financing_only_when_long(self):
        _, detail = gc.run_scenario(self.prices, leverage=2.0)
        long_days = detail[detail["position"] > 0.0]
        flat_days = detail[detail["position"] == 0.0]
        if len(long_days):
            self.assertTrue((long_days["leverage_financing"] > 0.0).all())
        self.assertTrue((flat_days["leverage_financing"] == 0.0).all())

    def test_flat_days_with_no_turnover_earn_exactly_zero(self):
        _, detail = gc.run_scenario(self.prices, leverage=1.0)
        flat_no_switch = detail[(detail["position"] == 0.0) & (detail["turnover"] == 0.0)]
        self.assertTrue((flat_no_switch["net_return"] == 0.0).all())

    def test_raises_on_empty_prices(self):
        empty = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            gc.run_scenario(empty, leverage=1.0)


if __name__ == "__main__":
    unittest.main()
