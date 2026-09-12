import unittest

import numpy as np
import pandas as pd

import bab_oos


def _prices_from_returns(dates, returns, start=100.0):
    return pd.Series(start * np.cumprod(1.0 + np.asarray(returns)), index=dates)


class ValidateTripleTests(unittest.TestCase):
    def _dates(self, n):
        return pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    def test_rejects_duplicate_dates(self):
        dates = self._dates(5)
        low = pd.Series([1.0] * 5, index=list(dates[:-1]) + [dates[-2]])
        high = pd.Series([1.0] * 5, index=dates)
        market = pd.Series([1.0] * 5, index=dates)
        with self.assertRaises(ValueError):
            bab_oos.validate_triple(low, high, market)

    def test_rejects_non_positive_prices(self):
        dates = self._dates(5)
        low = pd.Series([1.0, 1.0, 0.0, 1.0, 1.0], index=dates)
        high = pd.Series([1.0] * 5, index=dates)
        market = pd.Series([1.0] * 5, index=dates)
        with self.assertRaises(ValueError):
            bab_oos.validate_triple(low, high, market)

    def test_rejects_abnormally_long_gap(self):
        dates = list(self._dates(3)) + [pd.Timestamp("2021-06-01", tz="UTC")]
        series = pd.Series([1.0] * 4, index=dates)
        with self.assertRaises(ValueError):
            bab_oos.validate_triple(series, series, series)

    def test_keeps_only_overlapping_dates(self):
        dates = self._dates(5)
        low = pd.Series([1.0] * 5, index=dates)
        high = pd.Series([1.0] * 3, index=dates[:3])
        market = pd.Series([1.0] * 5, index=dates)
        aligned = bab_oos.validate_triple(low, high, market)
        self.assertEqual(len(aligned), 3)


class CausalRollingBetaTests(unittest.TestCase):
    def test_beta_excludes_current_day_return(self):
        dates = pd.bdate_range("2020-01-01", periods=300, tz="UTC")
        rng = np.random.default_rng(3)
        market = pd.Series(rng.normal(0.0, 0.01, 300), index=dates)
        asset = pd.Series(rng.normal(0.0, 0.01, 300), index=dates)

        beta = bab_oos.causal_rolling_beta(asset, market, window=252)
        probe_index = 280
        # 把当天(probe_index)的收益改成一个巨大离群值，如果beta真的
        # 是因果的（用shift(1)排除当天），这个改动不应该影响当天的
        # beta估计值
        mutated_market = market.copy()
        mutated_asset = asset.copy()
        mutated_market.iloc[probe_index] = 5.0
        mutated_asset.iloc[probe_index] = -5.0
        mutated_beta = bab_oos.causal_rolling_beta(
            mutated_asset, mutated_market, window=252
        )
        self.assertAlmostEqual(
            beta.iloc[probe_index], mutated_beta.iloc[probe_index], places=10
        )

    def test_recovers_approximate_known_beta(self):
        dates = pd.bdate_range("2020-01-01", periods=400, tz="UTC")
        rng = np.random.default_rng(5)
        market = pd.Series(rng.normal(0.0003, 0.012, 400), index=dates)
        idiosyncratic = rng.normal(0.0, 0.004, 400)
        asset = 0.5 * market + idiosyncratic
        beta = bab_oos.causal_rolling_beta(asset, market, window=252)
        self.assertAlmostEqual(beta.iloc[-1], 0.5, delta=0.15)


class RunScenarioTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2020-01-01", periods=400, tz="UTC")
        rng = np.random.default_rng(9)
        market_returns = rng.normal(0.0003, 0.012, 400)
        low_returns = 0.5 * market_returns + rng.normal(0.0, 0.004, 400)
        high_returns = 1.5 * market_returns + rng.normal(0.0, 0.004, 400)
        self.low = _prices_from_returns(self.dates, low_returns)
        self.high = _prices_from_returns(self.dates, high_returns)
        self.market = _prices_from_returns(self.dates, market_returns)

    def test_no_position_before_beta_window_is_full(self):
        _, detail = bab_oos.run_scenario(
            self.low, self.high, self.market, leverage=1.0, beta_window=252
        )
        early = detail.iloc[:250]
        self.assertTrue((early["weight_low"] == 0.0).all())
        self.assertTrue((early["weight_high"] == 0.0).all())

    def test_position_appears_once_beta_window_is_full(self):
        _, detail = bab_oos.run_scenario(
            self.low, self.high, self.market, leverage=1.0, beta_window=252
        )
        late = detail.iloc[300:]
        self.assertTrue((late["weight_low"] != 0.0).any())

    def test_portfolio_is_approximately_beta_neutral_at_rebalance(self):
        _, detail = bab_oos.run_scenario(
            self.low, self.high, self.market, leverage=1.0, beta_window=252
        )
        valid = detail.iloc[300:].copy()
        month_keys = valid.index.year * 100 + valid.index.month
        is_new_month = pd.Series(month_keys, index=valid.index).ne(
            pd.Series(month_keys, index=valid.index).shift(1)
        )
        rebalance_rows = valid[is_new_month.to_numpy()]
        self.assertGreater(len(rebalance_rows), 0)
        for _, row in rebalance_rows.iterrows():
            if pd.isna(row["beta_low"]) or pd.isna(row["beta_high"]):
                continue
            implied_beta = (
                row["weight_low"] * row["beta_low"]
                + row["weight_high"] * row["beta_high"]
            )
            # 权重是漂移后的收盘权重，不是再平衡目标本身，允许一天的
            # 价格漂移误差，但应该远小于1.0（贝塔中性核心特征）
            self.assertLess(abs(implied_beta), 0.1)

    def test_turnover_only_on_rebalance_days_once_beta_valid(self):
        _, detail = bab_oos.run_scenario(
            self.low, self.high, self.market, leverage=1.0, beta_window=252
        )
        valid = detail.iloc[300:]
        month_keys = pd.Series(
            valid.index.year * 100 + valid.index.month, index=valid.index
        )
        is_new_month = month_keys.ne(month_keys.shift(1))
        self.assertTrue((valid["turnover"][~is_new_month] == 0).all())

    def test_leverage_scales_target_weights(self):
        _, detail_1x = bab_oos.run_scenario(
            self.low, self.high, self.market, leverage=1.0, beta_window=252
        )
        _, detail_2x = bab_oos.run_scenario(
            self.low, self.high, self.market, leverage=2.0, beta_window=252
        )
        last_row_1x = detail_1x.iloc[-1]
        last_row_2x = detail_2x.iloc[-1]
        self.assertGreater(
            abs(last_row_2x["gross_exposure"]), abs(last_row_1x["gross_exposure"])
        )

    def test_raises_on_empty_alignment(self):
        empty = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            bab_oos.run_scenario(empty, self.high, self.market, leverage=1.0)


if __name__ == "__main__":
    unittest.main()
