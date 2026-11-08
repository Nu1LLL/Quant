"""Synthetic checks of the frozen ETH fee burden source and account."""

from datetime import date, timedelta
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

import pandas as pd

from research.eth_defi_tvl_source import SourceFailure
from research import eth_fee_burden_source as source
from research.eth_fee_burden_sim import simulate, weekly_target


class EthFeeBurdenTest(unittest.TestCase):
    def test_parser_requires_pinned_bytes_and_exact_day_schema(self):
        rows = [{"asset": "eth", "time": f"2020-11-{d:02d}T00:00:00.000000000Z",
                 "FeeTotNtv": "1.5", "TxCnt": "100"} for d in (1, 2)]
        body = json.dumps({"data": rows}, separators=(",", ":")).encode()
        with patch.object(source, "COUNT", 2), patch.object(source, "METRIC_SHA", sha256(body).hexdigest()):
            fees, tx = source.parse_metrics(body)
            self.assertEqual((fees[date(2020, 11, 2)], tx[date(2020, 11, 2)]), (1.5, 100))
            with self.assertRaisesRegex(SourceFailure, "response changed"):
                source.parse_metrics(body + b" ")
        rows[1]["TxCnt"] = "0"
        invalid = json.dumps({"data": rows}, separators=(",", ":")).encode()
        with patch.object(source, "COUNT", 2), patch.object(source, "METRIC_SHA", sha256(invalid).hexdigest()):
            with self.assertRaisesRegex(SourceFailure, "nonpositive"):
                source.parse_metrics(invalid)

    def test_monday_uses_only_friday_and_56_fixed_days(self):
        monday = date(2021, 1, 4)
        days = [monday - timedelta(days=i) for i in range(3, 59)]
        fees = {day: (2.0 if i <= 30 else 1.0)
                for i, day in enumerate(days, 3)}
        tx = {day: 100 for day in days}
        self.assertEqual(weekly_target(monday, fees, tx), 1)
        fees[monday - timedelta(days=1)] = 10_000_000.0
        fees[monday] = 10_000_000.0
        self.assertEqual(weekly_target(monday, fees, tx), 1)
        fees[days[0]] = 0.0
        tx[days[0]] = 10_000
        self.assertEqual(weekly_target(monday, fees, tx), 0)
        fees[days[0]] = 0
        del tx[days[-1]]
        with self.assertRaisesRegex(SourceFailure, "missing completed"):
            weekly_target(monday, fees, tx)

    def test_spot_replay_trades_first_monday_and_terminal(self):
        days = pd.date_range("2020-11-01", "2025-12-31", tz="UTC", freq="D")
        prices = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
                               "close": 100., "volume": 1_000_000.}, index=days)
        fees = {day.date(): float(i + 1) for i, day in enumerate(days)}
        tx = {day.date(): 100 for day in days}
        result = simulate(fees, tx, prices, 100.)
        self.assertEqual([(f["day"], f["side"]) for f in result.fills],
                         [("2021-01-04", "buy"), ("2026-01-01", "sell")])
        self.assertEqual(len(result.days), 1826)
        self.assertLess(result.terminal_cash, 10_000.)
        self.assertAlmostEqual(result.days[-1].end, result.terminal_cash)
        delayed = simulate(fees, tx, prices, 100., latency_days=1)
        self.assertEqual(delayed.fills[0]["day"], "2021-01-05")


if __name__ == "__main__":
    unittest.main()
