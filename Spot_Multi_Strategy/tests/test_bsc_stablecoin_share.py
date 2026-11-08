"""Synthetic BSC chain-share source, decision lag and BNB account tests."""

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

from research import bsc_stablecoin_share_source as source
from research.bsc_stablecoin_share_sim import simulate, weekly_target
from research.stablecoin_liquidity_source import LiquiditySourceFailure


class BscStablecoinShareTest(unittest.TestCase):
    def test_chain_chart_requires_pinned_subset_and_grid(self):
        first = date(2020, 12, 1)
        start = int(datetime(2020, 12, 1, tzinfo=timezone.utc).timestamp())
        rows = [{"date": str(start + i * 86_400),
                 "totalCirculating": {"peggedUSD": 100 + i},
                 "totalCirculatingUSD": {"peggedUSD": 100 + i}} for i in range(2)]
        body = json.dumps(rows).encode()
        pin = sha256(json.dumps([[start, 100], [start + 86_400, 101]],
                                separators=(",", ":")).encode()).hexdigest()
        with patch.object(source, "COUNT", 2), patch.object(source, "LAST", first + timedelta(days=1)):
            result = source.parse_chart(body, "BSC", pin)
            self.assertEqual(result[first + timedelta(days=1)], 101.0)
            with self.assertRaisesRegex(LiquiditySourceFailure, "changed"):
                source.parse_chart(body, "BSC", "0" * 64)

    def test_wednesday_sees_sunday_but_not_monday_or_tuesday(self):
        wednesday = date(2021, 1, 6)
        sunday = wednesday - timedelta(days=3)
        share = {sunday - timedelta(days=28): .1, sunday: .2,
                 sunday + timedelta(days=1): .01,
                 sunday + timedelta(days=2): .01}
        self.assertEqual(weekly_target(wednesday, share), 1)
        share[sunday] = .05
        self.assertEqual(weekly_target(wednesday, share), 0)
        del share[sunday]
        with self.assertRaisesRegex(LiquiditySourceFailure, "missing completed"):
            weekly_target(wednesday, share)

    def test_bnb_spot_replay_uses_thursday_and_terminal_open(self):
        days = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
        prices = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
                               "close": 100., "quote_volume": 10_000_000.}, index=days)
        share = {stamp.date(): .1 + i * .0001 for i, stamp in enumerate(days)}
        result = simulate(share, prices, 100.)
        self.assertEqual([(x["day"], x["side"]) for x in result.fills],
                         [("2021-01-07", "buy"), ("2026-01-01", "sell")])
        self.assertEqual(len(result.days), 1826)
        self.assertAlmostEqual(result.days[-1].end, result.terminal_cash)
        self.assertLess(result.terminal_cash, 10_000.)
        delayed = simulate(share, prices, 100., latency_days=1)
        self.assertEqual(delayed.fills[0]["day"], "2021-01-08")

    def test_terminal_grid_requires_all_31_timestamps(self):
        start = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000

        def archive(count: int) -> bytes:
            output = BytesIO()
            with ZipFile(output, "w") as zf:
                lines = []
                for i in range(count):
                    stamp = start + i * 86_400 * 1_000_000
                    row = [str(stamp), "100", "101", "99", "100", "1",
                           str(stamp + 86_400 * 1_000_000 - 1), "1000", "1", "1", "1000", "0"]
                    lines.append(",".join(row))
                zf.writestr("BNBUSDT-1d-2026-01.csv", "\n".join(lines) + "\n")
            return output.getvalue()

        full = archive(31)
        with patch.object(source, "_fetch", return_value=full):
            self.assertEqual(source._terminal_grid({"month": "2026-01",
                                                     "sha256": sha256(full).hexdigest()}), 31)
        short = archive(30)
        with patch.object(source, "_fetch", return_value=short):
            with self.assertRaisesRegex(LiquiditySourceFailure, "day count"):
                source._terminal_grid({"month": "2026-01",
                                       "sha256": sha256(short).hexdigest()})


if __name__ == "__main__":
    unittest.main()
