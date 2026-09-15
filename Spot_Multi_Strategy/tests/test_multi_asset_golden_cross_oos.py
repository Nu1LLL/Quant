import unittest

import numpy as np
import pandas as pd

import multi_asset_golden_cross_oos as mag


def _prices(dates, returns, start=100.0):
    return pd.Series(start * np.cumprod(1.0 + np.asarray(returns)), index=dates)


class AlignSleevesTests(unittest.TestCase):
    def test_keeps_only_common_dates(self):
        dates_long = pd.bdate_range("2020-01-01", periods=10, tz="UTC")
        dates_short = pd.bdate_range("2020-01-01", periods=5, tz="UTC")
        prices = {
            "QQQ": pd.Series(np.arange(10) + 100.0, index=dates_long),
            "FXE": pd.Series(np.arange(10) + 50.0, index=dates_long),
            "SLV": pd.Series(np.arange(10) + 20.0, index=dates_long),
            "ETHUSDT": pd.Series(np.arange(5) + 2000.0, index=dates_short),
        }
        aligned = mag.align_sleeves(prices)
        self.assertEqual(len(aligned), 5)
        self.assertListEqual(sorted(aligned.columns), ["ETHUSDT", "FXE", "QQQ", "SLV"])

    def test_aligns_across_different_intraday_timestamps_on_same_calendar_date(self):
        # Yahoo数据带交易所开盘时间戳(如13:30 UTC)，Binance数据是
        # 00:00 UTC——同一个交易日但精确时间戳不同，必须按日历日期
        # 对齐，不能按精确时间戳做内连接（早期实现在这里有真实bug）
        equity_index = pd.DatetimeIndex([
            pd.Timestamp("2020-01-02 13:30:00", tz="UTC"),
            pd.Timestamp("2020-01-03 13:30:00", tz="UTC"),
        ])
        crypto_index = pd.DatetimeIndex([
            pd.Timestamp("2020-01-02 00:00:00", tz="UTC"),
            pd.Timestamp("2020-01-03 00:00:00", tz="UTC"),
        ])
        prices = {
            "QQQ": pd.Series([100.0, 101.0], index=equity_index),
            "FXE": pd.Series([50.0, 50.5], index=equity_index),
            "SLV": pd.Series([20.0, 20.2], index=equity_index),
            "ETHUSDT": pd.Series([2000.0, 2010.0], index=crypto_index),
        }
        aligned = mag.align_sleeves(prices)
        self.assertEqual(len(aligned), 2)

    def test_raises_on_no_overlap(self):
        prices = {
            "QQQ": pd.Series([1.0], index=[pd.Timestamp("2020-01-01", tz="UTC")]),
            "FXE": pd.Series([1.0], index=[pd.Timestamp("2021-01-01", tz="UTC")]),
        }
        with self.assertRaises(ValueError):
            mag.align_sleeves(prices)


class RunSleeveReturnsTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2020-01-01", periods=260, tz="UTC")
        rng = np.random.default_rng(263)
        prices = {
            "QQQ": _prices(dates, rng.normal(0.0004, 0.011, 260)),
            "FXE": _prices(dates, rng.normal(0.0001, 0.005, 260)),
            "SLV": _prices(dates, rng.normal(0.0003, 0.018, 260)),
            "ETHUSDT": _prices(dates, rng.normal(0.0006, 0.03, 260)),
        }
        self.aligned = mag.align_sleeves(prices)

    def test_returns_one_series_per_sleeve(self):
        sleeve_returns, sleeve_details = mag.run_sleeve_returns(self.aligned, leverage=1.0)
        self.assertEqual(set(sleeve_returns.keys()), {"QQQ", "FXE", "SLV", "ETHUSDT"})
        self.assertEqual(set(sleeve_details.keys()), {"QQQ", "FXE", "SLV", "ETHUSDT"})

    def test_crypto_sleeve_uses_higher_cost_than_equity_sleeve(self):
        _, sleeve_details = mag.run_sleeve_returns(self.aligned, leverage=1.0)
        equity_turnover_cost = (
            sleeve_details["QQQ"]["turnover"] * mag.SLEEVE_LEG_COSTS["QQQ"]
        ).sum()
        crypto_turnover_cost = (
            sleeve_details["ETHUSDT"]["turnover"] * mag.SLEEVE_LEG_COSTS["ETHUSDT"]
        ).sum()
        equity_turnover = sleeve_details["QQQ"]["turnover"].sum()
        crypto_turnover = sleeve_details["ETHUSDT"]["turnover"].sum()
        if equity_turnover > 0 and crypto_turnover > 0:
            self.assertGreater(
                crypto_turnover_cost / crypto_turnover,
                equity_turnover_cost / equity_turnover,
            )


class CombineEqualWeightTests(unittest.TestCase):
    def test_combined_is_quarter_weighted_sum(self):
        dates = pd.bdate_range("2020-01-01", periods=5, tz="UTC")
        sleeve_returns = {
            "QQQ": pd.Series([0.01, 0.0, -0.02, 0.03, 0.0], index=dates),
            "FXE": pd.Series([0.0, 0.01, 0.0, -0.01, 0.02], index=dates),
            "SLV": pd.Series([-0.01, 0.02, 0.01, 0.0, -0.02], index=dates),
            "ETHUSDT": pd.Series([0.02, -0.01, 0.0, 0.01, 0.01], index=dates),
        }
        combined = mag.combine_equal_weight(sleeve_returns)
        expected = 0.25 * (
            sleeve_returns["QQQ"] + sleeve_returns["FXE"]
            + sleeve_returns["SLV"] + sleeve_returns["ETHUSDT"]
        )
        pd.testing.assert_series_equal(
            combined, expected.rename("net_return"), check_names=True
        )

    def test_all_flat_sleeves_produce_zero_combined_return(self):
        dates = pd.bdate_range("2020-01-01", periods=3, tz="UTC")
        zero = pd.Series([0.0, 0.0, 0.0], index=dates)
        sleeve_returns = {"QQQ": zero, "FXE": zero, "SLV": zero, "ETHUSDT": zero}
        combined = mag.combine_equal_weight(sleeve_returns)
        self.assertTrue((combined == 0.0).all())


if __name__ == "__main__":
    unittest.main()
