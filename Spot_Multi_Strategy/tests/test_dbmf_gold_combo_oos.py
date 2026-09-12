import unittest

import numpy as np
import pandas as pd

import dbmf_gold_combo_oos as combo


class AlignNetReturnsTests(unittest.TestCase):
    def test_keeps_only_overlapping_dates(self):
        dates_a = pd.bdate_range("2020-01-01", periods=5, tz="UTC")
        dates_b = pd.bdate_range("2020-01-01", periods=3, tz="UTC")
        a = pd.Series([0.01] * 5, index=dates_a)
        b = pd.Series([0.02] * 3, index=dates_b)
        aligned = combo.align_net_returns(a, b)
        self.assertEqual(len(aligned), 3)

    def test_raises_on_no_overlap(self):
        a = pd.Series([0.01], index=[pd.Timestamp("2020-01-01", tz="UTC")])
        b = pd.Series([0.01], index=[pd.Timestamp("2021-01-01", tz="UTC")])
        with self.assertRaises(ValueError):
            combo.align_net_returns(a, b)


class CombineEqualWeightTests(unittest.TestCase):
    def test_combined_return_is_arithmetic_average(self):
        dates = pd.bdate_range("2020-01-01", periods=5, tz="UTC")
        dbmf = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=dates)
        gold = pd.Series([0.02, 0.01, -0.01, 0.02, -0.03], index=dates)
        combined, aligned = combo.combine_equal_weight(dbmf, gold)
        expected = 0.5 * dbmf + 0.5 * gold
        pd.testing.assert_series_equal(
            combined, expected.rename("net_return"), check_names=True
        )
        self.assertEqual(len(aligned), 5)

    def test_no_extra_cost_is_added(self):
        # 两条输入序列本身就是已经扣过成本的"净"收益，组合后不应该
        # 再额外产生任何成本——如果两边完全相同，组合应该等于原值，
        # 不应该比任何一边更差
        dates = pd.bdate_range("2020-01-01", periods=10, tz="UTC")
        rng = np.random.default_rng(53)
        shared = pd.Series(rng.normal(0.001, 0.01, 10), index=dates)
        combined, _ = combo.combine_equal_weight(shared, shared)
        pd.testing.assert_series_equal(
            combined, shared.rename("net_return"), check_names=True
        )

    def test_diversification_reduces_volatility_when_legs_are_anti_correlated(self):
        dates = pd.bdate_range("2020-01-01", periods=100, tz="UTC")
        rng = np.random.default_rng(59)
        noise = rng.normal(0.0, 0.01, 100)
        dbmf = pd.Series(0.0002 + noise, index=dates)
        gold = pd.Series(0.0002 - noise, index=dates)
        combined, _ = combo.combine_equal_weight(dbmf, gold)
        self.assertLess(combined.std(), dbmf.std())
        self.assertLess(combined.std(), gold.std())


class PearsonCorrelationTests(unittest.TestCase):
    def test_perfectly_correlated_series_returns_one(self):
        dates = pd.bdate_range("2020-01-01", periods=50, tz="UTC")
        rng = np.random.default_rng(61)
        base = pd.Series(rng.normal(0.0, 0.01, 50), index=dates)
        correlation = combo.pearson_correlation(base, base * 2.0 + 0.001)
        self.assertAlmostEqual(correlation, 1.0, places=6)

    def test_perfectly_anti_correlated_series_returns_negative_one(self):
        dates = pd.bdate_range("2020-01-01", periods=50, tz="UTC")
        rng = np.random.default_rng(67)
        base = pd.Series(rng.normal(0.0, 0.01, 50), index=dates)
        correlation = combo.pearson_correlation(base, -base)
        self.assertAlmostEqual(correlation, -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
