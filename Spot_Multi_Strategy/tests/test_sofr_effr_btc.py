"""Synthetic rate, timing and accounting checks before opening live values."""

from __future__ import annotations

from datetime import date, timedelta
from hashlib import sha256
from unittest.mock import patch
import unittest

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure
from research.sofr_effr_source import parse_rate
from research.sofr_effr_sim import _target, event_schedule, simulate


def synthetic_rates():
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="B")
    sofr = {}
    effr = {}
    for i, stamp in enumerate(dates):
        day = stamp.date()
        effr[day] = 1.0
        sofr[day] = 1.0 + (.2 if (i // 20) % 2 else 0.0)
    return sofr, effr


class RepoStressTests(unittest.TestCase):
    def test_rate_parser_pins_header_dates_values_and_hash(self):
        body = b"observation_date,SOFR\n2020-01-01,1.0\n2020-01-02,1.1\n2021-01-01,1.2\n"
        spec = {"sha": sha256(body).hexdigest(), "bytes": len(body),
                "rows": 3, "first": date(2020, 1, 1),
                "last": date(2021, 1, 1), "counts": {2020: 2, 2021: 1}}
        with patch.dict("research.sofr_effr_source.RATE_SPECS", {"SOFR": spec}):
            rates = parse_rate("SOFR", body)
            self.assertEqual(len(rates), 3)
            with self.assertRaises(SourceFailure):
                parse_rate("SOFR", body.replace(b"2020-01-02", b"2020-01-01"))
            with self.assertRaises(SourceFailure):
                parse_rate("SOFR", body.replace(b"1.1", b"nan"))

    def test_weekly_rule_uses_only_twenty_dates_before_seven_day_cutoff(self):
        monday = date(2021, 2, 1)
        cutoff = monday - timedelta(days=7)
        dates = [stamp.date() for stamp in pd.date_range(end=cutoff, periods=20, freq="B")]
        sofr = {d: (2.0 if i < 10 else 1.0) for i, d in enumerate(dates)}
        effr = {d: 1.0 for d in dates}
        self.assertEqual(_target(monday, sofr, effr), 1)
        sofr[dates[-1] + timedelta(days=1)] = 100.0  # after cutoff: ignored
        effr[dates[-1] + timedelta(days=1)] = 1.0
        self.assertEqual(_target(monday, sofr, effr), 1)
        reverse = {d: (1.0 if i < 10 else 2.0) for i, d in enumerate(dates)}
        self.assertEqual(_target(monday, reverse, effr), 0)
        self.assertIsNone(_target(monday, {}, {}))

    def test_flat_price_replay_charges_cost_and_reconciles(self):
        sofr, effr = synthetic_rates()
        schedule, missing = event_schedule(sofr, effr, 0)
        self.assertEqual((len(schedule), missing), (261, ()))
        days = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
        prices = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                               "close": 100.0, "quote_volume": 2_000_000.0},
                              index=days)
        net = simulate(sofr, effr, prices, 100.0)
        gross = simulate(sofr, effr, prices, 100.0, slippage=0, fee_rate=0)
        self.assertEqual(len(net.days), 1826)
        self.assertGreater(len(net.fills), 0)
        self.assertLess(net.terminal_cash, 10_000)
        self.assertAlmostEqual(gross.terminal_cash, 10_000, places=6)
        self.assertAlmostEqual(net.days[-1].end, net.terminal_cash, places=7)


if __name__ == "__main__":
    unittest.main()
