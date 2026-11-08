"""Synthetic causality and accounting checks; never open formal market values."""

from datetime import date, timedelta
from hashlib import sha256
from unittest.mock import patch
import unittest

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure
from research.us_weekly_claims_btc_source import FIRST_WEEK, WEEK_COUNT, parse_icsa
from research.us_weekly_claims_btc_sim import claims_target, signal_schedule, simulate


class WeeklyClaimsTests(unittest.TestCase):
    def test_source_requires_pinned_weekly_grid_and_exact_sha(self):
        lines = ["observation_date,ICSA"]
        for index in range(WEEK_COUNT):
            week = FIRST_WEEK + timedelta(weeks=index)
            lines.append(f"{week},100000")
        body = ("\n".join(lines) + "\n").encode()
        with self.assertRaisesRegex(SourceFailure, "SHA changed"):
            parse_icsa(body)
        with patch("research.us_weekly_claims_btc_source.ICSA_SHA", sha256(body).hexdigest()):
            self.assertEqual(len(parse_icsa(body)), WEEK_COUNT)
        lines[-2] = "2026-10-23,100000"
        corrupt = ("\n".join(lines) + "\n").encode()
        with patch("research.us_weekly_claims_btc_source.ICSA_SHA", sha256(corrupt).hexdigest()):
            with self.assertRaisesRegex(SourceFailure, "date grid"):
                parse_icsa(corrupt)

    def test_strict_improvement_and_post_release_monday(self):
        week = date(2020, 12, 26)
        claims = {week - timedelta(weeks=k): 200000 for k in range(17)}
        self.assertEqual(claims_target(week, claims), 0)
        for k in range(4):
            claims[week - timedelta(weeks=k)] = 100000
        self.assertEqual(claims_target(week, claims), 1)
        for k in range(4):
            claims[week - timedelta(weeks=k)] = 200000
        self.assertEqual(claims_target(week, claims), 0)

        all_claims = {}
        day = date(2020, 8, 29)
        while day <= date(2025, 12, 27):
            all_claims[day] = 200000 if day < date(2020, 12, 5) else 100000
            day += timedelta(weeks=1)
        schedule = signal_schedule(all_claims)
        delayed = signal_schedule(all_claims, latency_weeks=1)
        self.assertEqual(schedule[date(2021, 1, 4)], 1)
        self.assertEqual(delayed[date(2021, 1, 11)], 1)
        self.assertEqual(delayed[date(2021, 1, 4)],
                         claims_target(date(2020, 12, 19), all_claims))

    def test_constant_price_loses_paid_costs_and_reconciles(self):
        claims = {}
        week = date(2020, 8, 29)
        while week <= date(2025, 12, 27):
            claims[week] = 200000 if week < date(2020, 12, 5) else 100000
            week += timedelta(weeks=1)
        dates = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
        prices = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                               "close": 100.0, "quote_volume": 100_000_000.0},
                              index=dates)
        result = simulate(claims, prices, 100.0)
        self.assertEqual(result.fills[0]["day"], "2021-01-04")
        self.assertEqual(result.fills[0]["side"], "buy")
        self.assertLess(result.terminal_cash, 10_000)
        self.assertGreater(result.fees, 0)
        self.assertAlmostEqual(result.days[-1].end, result.terminal_cash)
        self.assertTrue(all(fill["quantity"] <= 100 for fill in result.fills))


if __name__ == "__main__":
    unittest.main()
