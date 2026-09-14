import unittest

import numpy as np
import pandas as pd

import inflation_premium_oos as ip


def _prices(dates, protected_returns, nominal_returns, start=100.0):
    protected = start * np.cumprod(1.0 + np.asarray(protected_returns))
    nominal = start * np.cumprod(1.0 + np.asarray(nominal_returns))
    return (
        pd.Series(protected, index=dates),
        pd.Series(nominal, index=dates),
    )


class ValidatePairTests(unittest.TestCase):
    def _dates(self, n):
        return pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    def test_rejects_duplicate_dates(self):
        dates = self._dates(5)
        protected = pd.Series([1.0] * 5, index=list(dates[:-1]) + [dates[-2]])
        nominal = pd.Series([1.0] * 5, index=dates)
        with self.assertRaises(ValueError):
            ip.validate_pair(protected, nominal)

    def test_rejects_non_positive_prices(self):
        dates = self._dates(5)
        protected = pd.Series([1.0, 1.0, -1.0, 1.0, 1.0], index=dates)
        nominal = pd.Series([1.0] * 5, index=dates)
        with self.assertRaises(ValueError):
            ip.validate_pair(protected, nominal)

    def test_rejects_abnormally_long_gap(self):
        dates = list(self._dates(3)) + [pd.Timestamp("2021-06-01", tz="UTC")]
        series = pd.Series([1.0] * 4, index=dates)
        with self.assertRaises(ValueError):
            ip.validate_pair(series, series)

    def test_keeps_only_overlapping_dates(self):
        dates = self._dates(5)
        protected = pd.Series([1.0] * 5, index=dates)
        nominal = pd.Series([1.0] * 3, index=dates[:3])
        aligned = ip.validate_pair(protected, nominal)
        self.assertEqual(len(aligned), 3)


class RunScenarioTests(unittest.TestCase):
    def setUp(self):
        jan = pd.bdate_range("2020-01-01", "2020-01-31", tz="UTC")
        feb = pd.bdate_range("2020-02-01", "2020-02-29", tz="UTC")
        self.dates = jan.append(feb)
        rng = np.random.default_rng(139)
        self.protected_returns = rng.normal(0.0002, 0.004, len(self.dates))
        self.nominal_returns = rng.normal(0.0001, 0.0035, len(self.dates))
        self.protected, self.nominal = _prices(
            self.dates, self.protected_returns, self.nominal_returns
        )

    def test_turnover_only_on_rebalance_days(self):
        _, detail = ip.run_scenario(self.protected, self.nominal, leverage=1.0)
        month_keys = detail.index.year * 100 + detail.index.month
        is_new_month = month_keys != pd.Series(month_keys).shift(1).to_numpy()
        is_new_month[0] = True
        self.assertTrue((detail["turnover"][is_new_month] > 0).all())
        self.assertTrue((detail["turnover"][~is_new_month] == 0).all())

    def test_weights_reset_to_target_at_rebalance(self):
        _, detail = ip.run_scenario(self.protected, self.nominal, leverage=1.0)
        first_feb_day = detail.index[detail.index.month == 2][0]
        row = detail.loc[first_feb_day]
        expected_weight = 0.5 * (1 + row["inflation_protected_return"]) / (
            1 + row["net_return"]
        )
        self.assertAlmostEqual(row["weight_protected"], expected_weight, places=8)

    def test_weights_drift_away_from_target_between_rebalances(self):
        _, detail = ip.run_scenario(self.protected, self.nominal, leverage=1.0)
        jan_days = detail[detail.index.month == 1]
        last_jan_weight = jan_days["weight_protected"].iloc[-1]
        self.assertNotAlmostEqual(last_jan_weight, 0.5, places=4)

    def test_identical_legs_cancel_before_cost(self):
        dates = pd.bdate_range("2020-01-01", "2020-06-30", tz="UTC")
        rng = np.random.default_rng(149)
        shared_returns = rng.normal(0.0002, 0.004, len(dates))
        protected, nominal = _prices(dates, shared_returns, shared_returns)
        net, detail = ip.run_scenario(
            protected, nominal, leverage=1.0,
            annual_short_financing=0.0, annual_leverage_financing=0.0
        )
        total_cost = (detail["turnover"] * 0.0005).sum()
        self.assertAlmostEqual(net.sum(), -total_cost, places=2)

    def test_leverage_scales_financing_cost(self):
        _, detail_1x = ip.run_scenario(self.protected, self.nominal, leverage=1.0)
        _, detail_2x = ip.run_scenario(self.protected, self.nominal, leverage=2.0)
        cost_1x = (detail_1x["turnover"] * 0.0005 + 0.5 * 0.003 / 252.0).sum()
        cost_2x = (
            detail_2x["turnover"] * 0.0005
            + 1.0 * 0.003 / 252.0
            + 1.0 * 0.04 / 252.0
        ).sum()
        self.assertGreater(cost_2x, cost_1x)

    def test_raises_when_series_are_empty_after_alignment(self):
        empty = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            ip.run_scenario(empty, self.nominal, leverage=1.0)


if __name__ == "__main__":
    unittest.main()
