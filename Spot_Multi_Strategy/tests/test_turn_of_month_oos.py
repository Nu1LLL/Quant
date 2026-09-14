import unittest

import numpy as np
import pandas as pd

import turn_of_month_oos as tm


def _prices(dates, returns, start=100.0):
    return pd.Series(start * np.cumprod(1.0 + np.asarray(returns)), index=dates)


class ValidatePricesTests(unittest.TestCase):
    def _dates(self, n):
        return pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    def test_rejects_duplicate_dates(self):
        dates = self._dates(5)
        prices = pd.Series([1.0] * 5, index=list(dates[:-1]) + [dates[-2]])
        with self.assertRaises(ValueError):
            tm.validate_prices(prices)

    def test_rejects_non_positive_prices(self):
        dates = self._dates(5)
        prices = pd.Series([1.0, 1.0, -1.0, 1.0, 1.0], index=dates)
        with self.assertRaises(ValueError):
            tm.validate_prices(prices)

    def test_rejects_abnormally_long_gap(self):
        dates = list(self._dates(3)) + [pd.Timestamp("2021-06-01", tz="UTC")]
        prices = pd.Series([1.0] * 4, index=dates)
        with self.assertRaises(ValueError):
            tm.validate_prices(prices)


class BuildWindowFlagsTests(unittest.TestCase):
    def test_flags_first_three_and_last_trading_day(self):
        index = pd.bdate_range("2020-01-01", "2020-01-31", tz="UTC")
        flags = tm.build_window_flags(index)
        # 2020-01-01/02/03是1月前三个交易日，2020-01-31是1月最后一个交易日
        self.assertTrue(flags[0])
        self.assertTrue(flags[1])
        self.assertTrue(flags[2])
        self.assertFalse(flags[3])
        self.assertTrue(flags[-1])

    def test_middle_of_month_is_not_flagged(self):
        index = pd.bdate_range("2020-01-01", "2020-01-31", tz="UTC")
        flags = tm.build_window_flags(index)
        mid_month = index.get_loc(pd.Timestamp("2020-01-15", tz="UTC"))
        self.assertFalse(flags[mid_month])


class RunScenarioTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2020-01-01", "2020-06-30", tz="UTC")
        rng = np.random.default_rng(223)
        self.returns = rng.normal(0.0004, 0.011, len(self.dates))
        self.prices = _prices(self.dates, self.returns)

    def test_position_is_zero_or_leverage_only(self):
        _, detail = tm.run_scenario(self.prices, leverage=1.0)
        self.assertTrue(detail["position"].isin([0.0, 1.0]).all())

    def test_no_position_outside_window(self):
        _, detail = tm.run_scenario(self.prices, leverage=1.0)
        outside = detail[~detail["in_window"]]
        self.assertTrue((outside["position"] == 0.0).all())

    def test_flat_days_with_no_turnover_earn_exactly_zero(self):
        _, detail = tm.run_scenario(self.prices, leverage=1.0)
        outside = detail[~detail["in_window"]]
        non_switch = outside[outside["turnover"] == 0.0]
        self.assertTrue((non_switch["net_return"] == 0.0).all())

    def test_no_leverage_financing_at_1x(self):
        _, detail = tm.run_scenario(self.prices, leverage=1.0)
        self.assertTrue((detail["leverage_financing"] == 0.0).all())

    def test_leverage_financing_only_inside_window_above_1x(self):
        _, detail = tm.run_scenario(self.prices, leverage=2.0)
        inside = detail[detail["position"] > 0.0]
        outside = detail[detail["position"] == 0.0]
        self.assertTrue((inside["leverage_financing"] > 0.0).all())
        self.assertTrue((outside["leverage_financing"] == 0.0).all())

    def test_raises_on_empty_prices(self):
        empty = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            tm.run_scenario(empty, leverage=1.0)


if __name__ == "__main__":
    unittest.main()
