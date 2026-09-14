import unittest

import numpy as np
import pandas as pd

import january_effect_oos as je


def _prices(dates, small_returns, large_returns, start=100.0):
    small = start * np.cumprod(1.0 + np.asarray(small_returns))
    large = start * np.cumprod(1.0 + np.asarray(large_returns))
    return pd.Series(small, index=dates), pd.Series(large, index=dates)


class ValidatePairTests(unittest.TestCase):
    def _dates(self, n):
        return pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    def test_rejects_duplicate_dates(self):
        dates = self._dates(5)
        small = pd.Series([1.0] * 5, index=list(dates[:-1]) + [dates[-2]])
        large = pd.Series([1.0] * 5, index=dates)
        with self.assertRaises(ValueError):
            je.validate_pair(small, large)

    def test_rejects_non_positive_prices(self):
        dates = self._dates(5)
        small = pd.Series([1.0, 1.0, -1.0, 1.0, 1.0], index=dates)
        large = pd.Series([1.0] * 5, index=dates)
        with self.assertRaises(ValueError):
            je.validate_pair(small, large)

    def test_rejects_abnormally_long_gap(self):
        dates = list(self._dates(3)) + [pd.Timestamp("2021-06-01", tz="UTC")]
        series = pd.Series([1.0] * 4, index=dates)
        with self.assertRaises(ValueError):
            je.validate_pair(series, series)


class RunScenarioTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2019-11-01", "2020-04-30", tz="UTC")
        rng = np.random.default_rng(211)
        self.small_returns = rng.normal(0.0006, 0.014, len(self.dates))
        self.large_returns = rng.normal(0.0003, 0.011, len(self.dates))
        self.small, self.large = _prices(
            self.dates, self.small_returns, self.large_returns
        )

    def test_flat_outside_january(self):
        _, detail = je.run_scenario(self.small, self.large, leverage=1.0)
        non_january = detail[detail.index.month != 1]
        self.assertTrue((non_january["weight_small_cap"] == 0.0).all())
        self.assertTrue((non_january["weight_large_cap"] == 0.0).all())
        # 2月第一个交易日是平仓日，会产生一次性的换手成本（合理的
        # 负数），其余非1月交易日应该完全没有仓位、没有成本
        first_february = non_january.index[non_january.index.month == 2][0]
        other_non_january = non_january.drop(first_february)
        self.assertTrue((other_non_january["net_return"] == 0.0).all())
        self.assertLessEqual(detail.loc[first_february, "net_return"], 0.0)

    def test_position_active_within_january(self):
        _, detail = je.run_scenario(self.small, self.large, leverage=1.0)
        january = detail[detail.index.month == 1]
        self.assertTrue((january["weight_small_cap"] != 0.0).any())

    def test_turnover_at_entry_and_exit_only(self):
        _, detail = je.run_scenario(self.small, self.large, leverage=1.0)
        first_jan = detail.index[detail.index.month == 1][0]
        first_feb = detail.index[detail.index.month == 2][0]
        self.assertGreater(detail.loc[first_jan, "turnover"], 0.0)
        self.assertGreater(detail.loc[first_feb, "turnover"], 0.0)
        december_days = detail[detail.index.month == 12]
        self.assertTrue((december_days["turnover"] == 0.0).all())

    def test_no_cost_outside_january(self):
        _, detail = je.run_scenario(self.small, self.large, leverage=1.0)
        december_days = detail[detail.index.month == 12]
        self.assertTrue((december_days["net_return"] == 0.0).all())

    def test_leverage_scales_january_target(self):
        _, detail_1x = je.run_scenario(self.small, self.large, leverage=1.0)
        _, detail_2x = je.run_scenario(self.small, self.large, leverage=2.0)
        first_jan = detail_1x.index[detail_1x.index.month == 1][0]
        self.assertAlmostEqual(
            detail_2x.loc[first_jan, "turnover"],
            2.0 * detail_1x.loc[first_jan, "turnover"],
            places=8
        )

    def test_raises_on_empty_series(self):
        empty = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            je.run_scenario(empty, self.large, leverage=1.0)


if __name__ == "__main__":
    unittest.main()
