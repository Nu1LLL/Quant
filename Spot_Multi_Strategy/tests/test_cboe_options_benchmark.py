import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

import cboe_options_benchmark as options


class CboeDownloadTests(unittest.TestCase):
    def test_download_parses_official_csv_shape(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.text = "DATE,PUT\n01/03/2024,100.0\n01/04/2024,101.5\n"
        session = Mock()
        session.get.return_value = response
        series = options.download_index("PUT", session=session)
        np.testing.assert_allclose(series.to_numpy(), [100.0, 101.5])
        self.assertEqual(series.name, "PUT")

    def test_cache_round_trip(self):
        index = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
        source = pd.Series([100.0, 101.0, 102.0], index=index, name="PUT")
        with tempfile.TemporaryDirectory() as folder:
            with unittest.mock.patch.object(
                options, "download_index", return_value=source
            ):
                first = options.load_index("PUT", folder)
                second = options.load_index("PUT", folder)
        np.testing.assert_allclose(first, second)


class CboePortfolioTests(unittest.TestCase):
    def test_equal_weight_constant_assets_only_pays_initial_cost(self):
        index = pd.date_range("2024-01-01", periods=42, freq="B")
        levels = pd.DataFrame(
            100.0, index=index, columns=list(options.SYMBOLS)
        )
        returns, turnover = options.monthly_equal_weight_returns(levels)
        self.assertAlmostEqual(float(returns.iloc[0]), -0.0005)
        self.assertAlmostEqual(float(returns.iloc[1:].sum()), 0.0)
        self.assertAlmostEqual(float(turnover.sum()), 1.0)

    def test_leverage_deducts_incremental_financing(self):
        index = pd.date_range("2024-01-01", periods=2, freq="B")
        base = pd.Series([0.01, -0.01], index=index)
        result = options.apply_leverage(base, 2.0, annual_financing=0.04)
        expected = base * 2.0 - 0.04 / 252.0
        pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
