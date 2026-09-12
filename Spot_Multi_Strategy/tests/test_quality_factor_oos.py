import unittest

import numpy as np
import pandas as pd

import quality_factor_oos as qf


def _prices(dates, quality_returns, market_returns, start=100.0):
    quality = start * np.cumprod(1.0 + np.asarray(quality_returns))
    market = start * np.cumprod(1.0 + np.asarray(market_returns))
    return (
        pd.Series(quality, index=dates),
        pd.Series(market, index=dates),
    )


def _monthly_dates(periods):
    return pd.date_range("2020-01-01", periods=periods, freq="7D", tz="UTC")


class ValidatePairTests(unittest.TestCase):
    def test_rejects_duplicate_dates(self):
        dates = _monthly_dates(5)
        quality = pd.Series([1.0] * 5, index=list(dates[:-1]) + [dates[-2]])
        market = pd.Series([1.0] * 5, index=dates)
        with self.assertRaises(ValueError):
            qf.validate_pair(quality, market)

    def test_rejects_non_positive_prices(self):
        dates = _monthly_dates(5)
        quality = pd.Series([1.0, 1.0, -1.0, 1.0, 1.0], index=dates)
        market = pd.Series([1.0] * 5, index=dates)
        with self.assertRaises(ValueError):
            qf.validate_pair(quality, market)

    def test_rejects_abnormally_long_gap(self):
        dates = list(_monthly_dates(3)) + [pd.Timestamp("2021-06-01", tz="UTC")]
        quality = pd.Series([1.0] * 4, index=dates)
        market = pd.Series([1.0] * 4, index=dates)
        with self.assertRaises(ValueError):
            qf.validate_pair(quality, market)

    def test_keeps_only_overlapping_dates(self):
        dates = _monthly_dates(5)
        quality = pd.Series([1.0] * 5, index=dates)
        market = pd.Series([1.0] * 3, index=dates[:3])
        aligned = qf.validate_pair(quality, market)
        self.assertEqual(len(aligned), 3)


class RunScenarioTests(unittest.TestCase):
    def setUp(self):
        # 用日频数据，前两个月各21个交易日，制造清晰的月度边界
        jan = pd.bdate_range("2020-01-01", "2020-01-31", tz="UTC")
        feb = pd.bdate_range("2020-02-01", "2020-02-29", tz="UTC")
        self.dates = jan.append(feb)
        rng = np.random.default_rng(7)
        self.quality_returns = rng.normal(0.0006, 0.01, len(self.dates))
        self.market_returns = rng.normal(0.0003, 0.012, len(self.dates))
        self.quality, self.market = _prices(
            self.dates, self.quality_returns, self.market_returns
        )

    def test_turnover_only_on_rebalance_days(self):
        _, detail = qf.run_scenario(self.quality, self.market, leverage=1.0)
        month_keys = detail.index.year * 100 + detail.index.month
        is_new_month = month_keys != pd.Series(month_keys).shift(1).to_numpy()
        is_new_month[0] = True
        self.assertTrue((detail["turnover"][is_new_month] > 0).all())
        self.assertTrue((detail["turnover"][~is_new_month] == 0).all())

    def test_weights_reset_to_target_at_rebalance(self):
        _, detail = qf.run_scenario(self.quality, self.market, leverage=1.0)
        first_feb_day = detail.index[detail.index.month == 2][0]
        row = detail.loc[first_feb_day]
        # 再平衡日当天，权重从目标0.5出发只漂移了当天一次，
        # weight = start(=0.5) * (1+当天收益) / (1+当天净收益)
        expected_weight_quality = 0.5 * (1 + row["quality_return"]) / (
            1 + row["net_return"]
        )
        self.assertAlmostEqual(
            row["weight_quality"], expected_weight_quality, places=8
        )

    def test_weights_drift_away_from_target_between_rebalances(self):
        _, detail = qf.run_scenario(self.quality, self.market, leverage=1.0)
        jan_days = detail[detail.index.month == 1]
        last_jan_weight = jan_days["weight_quality"].iloc[-1]
        self.assertNotAlmostEqual(last_jan_weight, 0.5, places=4)

    def test_identical_legs_cancel_before_cost(self):
        # QUAL和SPY走势完全一样时，多空应该几乎完全对冲，净收益
        # 应该约等于成本的负值，不应该有系统性方向性收益
        dates = pd.bdate_range("2020-01-01", "2020-06-30", tz="UTC")
        rng = np.random.default_rng(11)
        shared_returns = rng.normal(0.0004, 0.011, len(dates))
        quality, market = _prices(dates, shared_returns, shared_returns)
        net, detail = qf.run_scenario(
            quality, market, leverage=1.0,
            annual_short_financing=0.0, annual_leverage_financing=0.0
        )
        total_cost = (detail["turnover"] * 0.0005).sum()
        self.assertAlmostEqual(
            net.sum(), -total_cost, places=2
        )

    def test_leverage_scales_financing_cost(self):
        _, detail_1x = qf.run_scenario(self.quality, self.market, leverage=1.0)
        _, detail_2x = qf.run_scenario(self.quality, self.market, leverage=2.0)
        cost_1x = (
            detail_1x["turnover"] * 0.0005
            + 0.5 * 0.003 / 252.0
        ).sum()
        cost_2x = (
            detail_2x["turnover"] * 0.0005
            + 1.0 * 0.003 / 252.0
            + 1.0 * 0.04 / 252.0
        ).sum()
        self.assertGreater(cost_2x, cost_1x)

    def test_raises_when_series_are_empty_after_alignment(self):
        empty = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            qf.run_scenario(empty, self.market, leverage=1.0)


if __name__ == "__main__":
    unittest.main()
