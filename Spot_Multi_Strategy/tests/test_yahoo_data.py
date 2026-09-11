import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pandas as pd

from yahoo_data import download_yahoo_daily_klines, load_or_download_yahoo_klines


def _fake_yahoo_payload(timestamps, opens, highs, lows, closes, volumes):
    return {
        "chart": {
            "result": [{
                "meta": {"symbol": "TEST"},
                "timestamp": timestamps,
                "indicators": {
                    "quote": [{
                        "open": opens,
                        "high": highs,
                        "low": lows,
                        "close": closes,
                        "volume": volumes
                    }]
                }
            }]
        }
    }


class YahooParsingTests(unittest.TestCase):
    """不联网：用mock的Session模拟Yahoo接口返回，和仓库其余测试
    "不联网的自动化测试"的约定保持一致（真正的网络请求不在单测里测）。
    """

    def _mock_session(self, payload):
        session = MagicMock()
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        session.get.return_value = response
        return session

    def test_parses_ohlcv_columns_correctly(self):
        payload = _fake_yahoo_payload(
            timestamps=[1700000000, 1700086400, 1700172800],
            opens=[100.0, 101.0, 102.0],
            highs=[102.0, 103.0, 104.0],
            lows=[99.0, 100.0, 101.0],
            closes=[101.0, 102.0, 103.0],
            volumes=[1000, 1100, 1200]
        )
        session = self._mock_session(payload)
        df = download_yahoo_daily_klines("TEST", session=session)

        self.assertEqual(len(df), 3)
        self.assertListEqual(
            list(df.columns),
            ["open_time", "open", "high", "low", "close", "volume"]
        )
        self.assertAlmostEqual(df.iloc[0]["close"], 101.0)
        self.assertIsInstance(df["open_time"].dtype, pd.DatetimeTZDtype)

    def test_rows_with_missing_prices_are_dropped_not_filled(self):
        payload = _fake_yahoo_payload(
            timestamps=[1700000000, 1700086400, 1700172800],
            opens=[100.0, None, 102.0],
            highs=[102.0, None, 104.0],
            lows=[99.0, None, 101.0],
            closes=[101.0, None, 103.0],
            volumes=[1000, None, 1200]
        )
        session = self._mock_session(payload)
        df = download_yahoo_daily_klines("TEST", session=session)

        self.assertEqual(len(df), 2)
        self.assertFalse(df["close"].isna().any())

    def test_missing_result_raises_instead_of_returning_empty_silently(self):
        session = self._mock_session({"chart": {"result": None}})
        with self.assertRaises(ValueError):
            download_yahoo_daily_klines("TEST", session=session)


class YahooCacheTests(unittest.TestCase):
    def test_reads_from_cache_without_calling_network_when_present(self):
        with TemporaryDirectory() as tmp_dir:
            cache_folder = Path(tmp_dir)
            cache_file = cache_folder / "TEST_10y_1d.csv"
            pd.DataFrame({
                "open_time": pd.to_datetime(
                    ["2021-01-01"], utc=True
                ),
                "open": [1.0], "high": [1.0],
                "low": [1.0], "close": [1.0], "volume": [1]
            }).to_csv(cache_file, index=False)

            df = load_or_download_yahoo_klines(
                "TEST", cache_folder=str(cache_folder)
            )
            self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
