import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

import cross_asset_trend as trend


class YahooAdjustedCloseTests(unittest.TestCase):
    def test_download_uses_adjusted_close(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "chart": {"result": [{
                "timestamp": [1704067200, 1704153600],
                "indicators": {
                    "adjclose": [{"adjclose": [99.0, 101.0]}],
                },
            }]}
        }
        session = Mock()
        session.get.return_value = response
        result = trend.download_adjusted_close(
            "TEST", "2024-01-01", "2024-01-03", session=session
        )
        np.testing.assert_allclose(result.to_numpy(), [99.0, 101.0])
        self.assertEqual(result.name, "TEST")

    def test_cache_round_trip(self):
        index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        with tempfile.TemporaryDirectory() as folder:
            with unittest.mock.patch.object(
                trend, "download_adjusted_close",
                return_value=pd.Series([1.0, 2.0, 3.0], index=index),
            ):
                first = trend.load_adjusted_close(
                    "TEST", "2024-01-01", "2024-01-04", folder
                )
                second = trend.load_adjusted_close(
                    "TEST", "2024-01-01", "2024-01-04", folder
                )
        np.testing.assert_allclose(first, second)


class CrossAssetTrendTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        index = pd.date_range("2020-01-01", periods=900, freq="B")
        returns = rng.normal(0.0002, 0.01, size=(900, 4))
        self.prices = pd.DataFrame(
            100 * np.exp(np.cumsum(returns, axis=0)),
            index=index, columns=list("ABCD")
        )

    def test_future_mutation_does_not_change_past_positions(self):
        before = trend.build_trend_positions(self.prices)
        changed = self.prices.copy()
        changed.iloc[700:] *= 3.0
        after = trend.build_trend_positions(changed)
        pd.testing.assert_frame_equal(before.iloc[:700], after.iloc[:700])

    def test_positions_respect_final_gross_cap(self):
        positions = trend.build_trend_positions(
            self.prices, leverage=100.0, gross_cap=4.0
        )
        self.assertLessEqual(
            float(positions.abs().sum(axis=1).max()), 4.0 + 1e-12
        )

    def test_backtest_costs_are_nonnegative(self):
        simulation, positions, yearly, metrics = trend.run_backtest(
            self.prices, leverage=2.0
        )
        self.assertTrue((simulation["cost"] >= 0).all())
        self.assertEqual(len(simulation), len(self.prices))
        self.assertIn("sharpe_ratio", metrics)


if __name__ == "__main__":
    unittest.main()
