import unittest

import numpy as np
import pandas as pd

import dbmf_oos as dbmf


class ValidatePricesTests(unittest.TestCase):
    def _dates(self, n):
        return pd.bdate_range("2020-01-01", periods=n, tz="UTC")

    def test_rejects_duplicate_dates(self):
        dates = self._dates(5)
        prices = pd.Series([1.0] * 5, index=list(dates[:-1]) + [dates[-2]])
        with self.assertRaises(ValueError):
            dbmf.validate_prices(prices)

    def test_rejects_non_positive_prices(self):
        dates = self._dates(5)
        prices = pd.Series([1.0, 1.0, 0.0, 1.0, 1.0], index=dates)
        with self.assertRaises(ValueError):
            dbmf.validate_prices(prices)

    def test_rejects_abnormally_long_gap(self):
        dates = list(self._dates(3)) + [pd.Timestamp("2021-06-01", tz="UTC")]
        prices = pd.Series([1.0] * 4, index=dates)
        with self.assertRaises(ValueError):
            dbmf.validate_prices(prices)

    def test_accepts_clean_series(self):
        dates = self._dates(5)
        prices = pd.Series([1.0, 1.01, 1.02, 1.01, 1.03], index=dates)
        validated = dbmf.validate_prices(prices)
        self.assertEqual(len(validated), 5)


class EvaluateFiveYearOosTests(unittest.TestCase):
    def test_uses_standard_five_year_gate_not_ten_or_fifteen(self):
        dates = pd.bdate_range("2019-01-01", periods=252 * 6, tz="UTC")
        rng = np.random.default_rng(23)
        returns = pd.Series(rng.normal(0.0003, 0.006, len(dates)), index=dates)
        result = dbmf.evaluate_five_year_oos(returns)
        self.assertIn("sample_at_least_5y", result["checks"])
        self.assertNotIn("sample_at_least_10y", result["checks"])
        self.assertNotIn("sample_at_least_15y", result["checks"])

    def test_independent_oos_and_costs_included_are_fixed_true(self):
        dates = pd.bdate_range("2019-01-01", periods=252 * 6, tz="UTC")
        rng = np.random.default_rng(29)
        returns = pd.Series(rng.normal(0.0002, 0.005, len(dates)), index=dates)
        result = dbmf.evaluate_five_year_oos(returns)
        self.assertTrue(result["checks"]["independent_oos"])
        self.assertTrue(result["checks"]["costs_included"])


if __name__ == "__main__":
    unittest.main()
