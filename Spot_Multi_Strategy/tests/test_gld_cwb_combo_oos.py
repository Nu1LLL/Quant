import unittest

import numpy as np
import pandas as pd

import gld_cwb_combo_oos as combo


class AlignNetReturnsTests(unittest.TestCase):
    def test_keeps_only_overlapping_dates(self):
        dates_a = pd.bdate_range("2020-01-01", periods=5, tz="UTC")
        dates_b = pd.bdate_range("2020-01-01", periods=3, tz="UTC")
        a = pd.Series([0.01] * 5, index=dates_a)
        b = pd.Series([0.02] * 3, index=dates_b)
        aligned = combo.align_net_returns(a, b)
        self.assertEqual(len(aligned), 3)
        self.assertListEqual(list(aligned.columns), ["GLD", "CWB"])

    def test_raises_on_no_overlap(self):
        a = pd.Series([0.01], index=[pd.Timestamp("2020-01-01", tz="UTC")])
        b = pd.Series([0.01], index=[pd.Timestamp("2021-01-01", tz="UTC")])
        with self.assertRaises(ValueError):
            combo.align_net_returns(a, b)


class CombineEqualWeightTests(unittest.TestCase):
    def test_combined_return_is_arithmetic_average(self):
        dates = pd.bdate_range("2020-01-01", periods=5, tz="UTC")
        gold = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=dates)
        cwb = pd.Series([0.02, 0.01, -0.01, 0.02, -0.03], index=dates)
        combined, aligned = combo.combine_equal_weight(gold, cwb)
        expected = 0.5 * gold + 0.5 * cwb
        pd.testing.assert_series_equal(
            combined, expected.rename("net_return"), check_names=True
        )
        self.assertEqual(len(aligned), 5)

    def test_no_extra_cost_is_added(self):
        dates = pd.bdate_range("2020-01-01", periods=10, tz="UTC")
        rng = np.random.default_rng(107)
        shared = pd.Series(rng.normal(0.001, 0.01, 10), index=dates)
        combined, _ = combo.combine_equal_weight(shared, shared)
        pd.testing.assert_series_equal(
            combined, shared.rename("net_return"), check_names=True
        )

    def test_diversification_reduces_volatility_when_legs_are_anti_correlated(self):
        dates = pd.bdate_range("2020-01-01", periods=100, tz="UTC")
        rng = np.random.default_rng(109)
        noise = rng.normal(0.0, 0.01, 100)
        gold = pd.Series(0.0002 + noise, index=dates)
        cwb = pd.Series(0.0002 - noise, index=dates)
        combined, _ = combo.combine_equal_weight(gold, cwb)
        self.assertLess(combined.std(), gold.std())
        self.assertLess(combined.std(), cwb.std())


class PearsonCorrelationTests(unittest.TestCase):
    def test_perfectly_correlated_series_returns_one(self):
        dates = pd.bdate_range("2020-01-01", periods=50, tz="UTC")
        rng = np.random.default_rng(113)
        base = pd.Series(rng.normal(0.0, 0.01, 50), index=dates)
        correlation = combo.pearson_correlation(base, base * 2.0 + 0.001)
        self.assertAlmostEqual(correlation, 1.0, places=6)

    def test_perfectly_anti_correlated_series_returns_negative_one(self):
        dates = pd.bdate_range("2020-01-01", periods=50, tz="UTC")
        rng = np.random.default_rng(127)
        base = pd.Series(rng.normal(0.0, 0.01, 50), index=dates)
        correlation = combo.pearson_correlation(base, -base)
        self.assertAlmostEqual(correlation, -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
