"""Synthetic source, as-of, and two-asset accounting checks only."""

from datetime import date, timedelta
from hashlib import sha256
import json
from unittest.mock import patch
import unittest

import pandas as pd

from research.btc_eth_transfer_growth_source import (
    METRIC_DAYS, METRIC_FIRST, SourceFailure, parse_transfer_counts,
)
from research.btc_eth_transfer_growth_sim import simulate, transfer_target


def counts() -> dict[str, dict[date, int]]:
    out = {"BTC": {}, "ETH": {}}
    for i in range(METRIC_DAYS):
        day = METRIC_FIRST + timedelta(days=i)
        out["BTC"][day] = 200 if date(2020, 12, 5) <= day <= date(2021, 1, 1) else 100
        out["ETH"][day] = 100
    return out


class TransferGrowthTests(unittest.TestCase):
    def test_metric_sha_grid_and_unique_asset_dates(self):
        rows = [{"asset": asset, "time": f"{METRIC_FIRST + timedelta(days=i)}T00:00:00.000000000Z",
                 "TxTfrCnt": "100"}
                for asset in ("btc", "eth") for i in range(METRIC_DAYS)]
        body = json.dumps({"data": rows}, separators=(",", ":")).encode()
        with self.assertRaisesRegex(SourceFailure, "SHA changed"):
            parse_transfer_counts(body)
        with patch("research.btc_eth_transfer_growth_source.CM_SHA", sha256(body).hexdigest()):
            result = parse_transfer_counts(body)
            self.assertEqual(len(result["BTC"]), METRIC_DAYS)
            self.assertEqual(len(result["ETH"]), METRIC_DAYS)
        rows[1]["time"] = rows[0]["time"]
        corrupt = json.dumps({"data": rows}, separators=(",", ":")).encode()
        with patch("research.btc_eth_transfer_growth_source.CM_SHA", sha256(corrupt).hexdigest()):
            with self.assertRaisesRegex(SourceFailure, "duplicated"):
                parse_transfer_counts(corrupt)

    def test_friday_cutoff_and_exact_relative_growth_comparison(self):
        data = counts()
        monday = date(2021, 1, 4)
        self.assertEqual(transfer_target(monday, data), "BTC")
        data["BTC"][date(2021, 1, 2)] = 9_999_999
        data["ETH"][date(2021, 1, 3)] = 9_999_999
        self.assertEqual(transfer_target(monday, data), "BTC")
        for asset in ("BTC", "ETH"):
            for day in data[asset]:
                data[asset][day] = 100
        self.assertIsNone(transfer_target(monday, data))

    def test_constant_price_switches_pay_costs_and_reconcile(self):
        data = counts()
        frames = {}
        for asset, first in (("BTC", "2020-12-01"), ("ETH", "2020-11-01")):
            dates = pd.date_range(first, "2025-12-31", tz="UTC", freq="D")
            frames[asset] = pd.DataFrame({
                "open": 100.0, "high": 100.0, "low": 100.0,
                "close": 100.0, "quote_volume": 100_000_000.0}, index=dates)
        result = simulate(data, frames, {"BTC": 100.0, "ETH": 100.0})
        self.assertEqual((result.fills[0]["day"], result.fills[0]["asset"],
                          result.fills[0]["side"]), ("2021-01-04", "BTC", "buy"))
        self.assertGreaterEqual(len(result.fills), 2)
        self.assertLess(result.terminal_cash, 10_000)
        self.assertGreater(result.fees, 0)
        self.assertAlmostEqual(result.days[-1].end, result.terminal_cash)
        self.assertEqual(len(result.days), 1826)
        static = simulate(data, frames, {"BTC": 100.0, "ETH": 100.0},
                          static_half_half=True)
        self.assertEqual([f["asset"] for f in static.fills[:2]], ["BTC", "ETH"])
        self.assertAlmostEqual(static.days[-1].end, static.terminal_cash)


if __name__ == "__main__":
    unittest.main()
