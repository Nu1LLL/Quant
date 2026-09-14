import unittest

import numpy as np
import pandas as pd

import halloween_seasonality_oos as hs


def _prices(dates, returns, start=100.0):
    return pd.Series(start * np.cumprod(1.0 + np.asarray(returns)), index=dates)


class ValidatePricesTests(unittest.TestCase):
    def _dates(self, n):
        return pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    def test_rejects_duplicate_dates(self):
        dates = self._dates(5)
        prices = pd.Series([1.0] * 5, index=list(dates[:-1]) + [dates[-2]])
        with self.assertRaises(ValueError):
            hs.validate_prices(prices)

    def test_rejects_non_positive_prices(self):
        dates = self._dates(5)
        prices = pd.Series([1.0, 1.0, -1.0, 1.0, 1.0], index=dates)
        with self.assertRaises(ValueError):
            hs.validate_prices(prices)

    def test_rejects_abnormally_long_gap(self):
        dates = list(self._dates(3)) + [pd.Timestamp("2021-06-01", tz="UTC")]
        prices = pd.Series([1.0] * 4, index=dates)
        with self.assertRaises(ValueError):
            hs.validate_prices(prices)


class BuildTargetPositionsTests(unittest.TestCase):
    def test_winter_months_are_long_summer_months_are_flat(self):
        index = pd.date_range("2020-01-01", "2020-12-31", freq="D", tz="UTC")
        target = hs.build_target_positions(index, leverage=1.0)
        for month in (11, 12, 1, 2, 3, 4):
            self.assertTrue((target[index.month == month] == 1.0).all())
        for month in (5, 6, 7, 8, 9, 10):
            self.assertTrue((target[index.month == month] == 0.0).all())

    def test_leverage_scales_winter_target_only(self):
        index = pd.date_range("2020-01-01", "2020-12-31", freq="D", tz="UTC")
        target = hs.build_target_positions(index, leverage=2.0)
        self.assertTrue((target[index.month == 1] == 2.0).all())
        self.assertTrue((target[index.month == 7] == 0.0).all())


class RunScenarioTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2019-06-01", "2021-06-30", tz="UTC")
        rng = np.random.default_rng(197)
        self.returns = rng.normal(0.0004, 0.011, len(self.dates))
        self.prices = _prices(self.dates, self.returns)

    def test_position_is_zero_or_leverage_only(self):
        _, detail = hs.run_scenario(self.prices, leverage=1.0)
        self.assertTrue(detail["position"].isin([0.0, 1.0]).all())

    def test_turnover_only_at_month_switch_points(self):
        _, detail = hs.run_scenario(self.prices, leverage=1.0)
        position_changed = detail["position"].diff().fillna(detail["position"].iloc[0]) != 0
        self.assertTrue((detail["turnover"][position_changed] > 0).all())
        self.assertTrue((detail["turnover"][~position_changed] == 0).all())

    def test_flat_months_earn_exactly_zero_before_cost(self):
        _, detail = hs.run_scenario(self.prices, leverage=1.0)
        summer_rows = detail[detail["position"] == 0.0]
        non_switch_summer = summer_rows[summer_rows["turnover"] == 0.0]
        self.assertTrue((non_switch_summer["net_return"] == 0.0).all())

    def test_no_leverage_financing_at_1x(self):
        _, detail = hs.run_scenario(self.prices, leverage=1.0)
        self.assertTrue((detail["leverage_financing"] == 0.0).all())

    def test_leverage_financing_only_during_winter_months_above_1x(self):
        _, detail = hs.run_scenario(self.prices, leverage=2.0)
        winter_rows = detail[detail["position"] > 0.0]
        summer_rows = detail[detail["position"] == 0.0]
        self.assertTrue((winter_rows["leverage_financing"] > 0.0).all())
        self.assertTrue((summer_rows["leverage_financing"] == 0.0).all())

    def test_raises_on_empty_prices(self):
        empty = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            hs.run_scenario(empty, leverage=1.0)


if __name__ == "__main__":
    unittest.main()
