"""Synthetic Treasury schema and coverage tests; no real balances are read."""

from datetime import date, timedelta
import json
import unittest

from research.tga_cash_source import (
    LAYOUT_CHANGE, TgaSourceError, YEAR_COUNTS, _balance_for_row, _parse,
)


URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance"


def _fixture():
    rows = []
    for year, wanted in YEAR_COUNTS.items():
        start = max(date(year, 1, 1), date(2020, 11, 1))
        end = min(date(year, 12, 31), date(2026, 9, 30))
        days = []
        day = start
        while day <= end:
            if day.weekday() < 5:
                days.append(day)
            day += timedelta(days=1)
        remove = len(days) - wanted
        Wednesdays = [day for day in days if day.weekday() == 2]
        removed = set(Wednesdays[:remove])
        for day in days:
            if day in removed:
                continue
            label = ("Federal Reserve Account" if day < LAYOUT_CHANGE else
                     "Treasury General Account (TGA) Closing Balance")
            rows.append({"record_date": day.isoformat(), "account_type": label,
                         "close_today_bal": "100000", "open_today_bal": "100000"})
    filler_count = 5566 - len(rows)
    rows.extend({"record_date": "2020-11-02",
                 "account_type": "Supplementary Financing Program Account",
                 "close_today_bal": "null", "open_today_bal": "null"}
                for _ in range(filler_count))
    return rows


def _payload(rows):
    return json.dumps({"meta": {"total-count": 5566, "count": 5566,
                                "total-pages": 1}, "data": rows}).encode()


class TgaCashSourceTests(unittest.TestCase):
    def test_frozen_two_schema_layouts_and_date_coverage(self):
        bundle = _parse(_payload(_fixture()), URL)
        self.assertEqual(len(bundle.observations), 1483)
        self.assertEqual(bundle.observations[0].observed_date, date(2020, 11, 2))
        self.assertEqual(bundle.observations[0].balance_million_usd, 100000)
        self.assertEqual(bundle.observations[0].proxy_available_at.date(),
                         date(2020, 11, 6))
        self.assertEqual(bundle.manifest["source_rows"], 5566)

    def test_daily_duplicate_or_missing_selected_balance_fails(self):
        rows = _fixture()
        rows[-1] = dict(rows[0])
        with self.assertRaisesRegex(TgaSourceError, "duplicate"):
            _parse(_payload(rows), URL)
        rows = _fixture()
        rows[0]["close_today_bal"] = "null"
        with self.assertRaisesRegex(TgaSourceError, "nonnumeric"):
            _parse(_payload(rows), URL)

    def test_unregistered_row_cannot_supply_a_balance(self):
        row = {"record_date": "2020-11-02", "account_type": "Total Operating Balance",
               "close_today_bal": "100", "open_today_bal": "100"}
        self.assertIsNone(_balance_for_row(row, date(2020, 11, 2)))
        row["account_type"] = "Treasury General Account (TGA) Closing Balance"
        self.assertEqual(_balance_for_row(row, date(2022, 4, 18)), 100)


if __name__ == "__main__":
    unittest.main()
