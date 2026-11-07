"""Synthetic Treasury CSV source checks; no real rates or market prices."""

from datetime import date, timedelta
import unittest

from research.tips_real_yield_source import (
    HEADER, URLS, TreasurySourceError, _parse_source, load_and_audit,
)


def _csv(rows: list[tuple[str, str]]) -> bytes:
    lines = [",".join(HEADER)]
    lines.extend(",".join((day, "0", "0", yield_value, "0", "0"))
                 for day, yield_value in rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _weekdays(first: date, last: date, *, two_digit: bool = False) -> list[tuple[str, str]]:
    rows = []
    day = first
    while day <= last:
        if day.weekday() < 5:
            rows.append((day.strftime("%m/%d/%y" if two_digit else "%m/%d/%Y"), "1.25"))
        day += timedelta(days=1)
    return rows[::-1]


class TreasuryRealYieldSourceTests(unittest.TestCase):
    def test_accepts_fixed_windows_and_never_parses_post_sample_numeric_value(self):
        bodies = (
            _csv(_weekdays(date(2020, 11, 1), date(2024, 12, 31), two_digit=True)),
            _csv(_weekdays(date(2025, 1, 1), date(2025, 12, 31))),
            _csv([("10/01/2026", "not-a-number")] +
                 _weekdays(date(2026, 1, 1), date(2026, 9, 30))),
        )
        supplied = dict(zip(URLS, bodies, strict=True))
        bundle = load_and_audit(fetcher=supplied.__getitem__)
        self.assertEqual(bundle.observations[0].observed_date, date(2020, 11, 2))
        self.assertEqual(bundle.observations[-1].observed_date, date(2026, 9, 30))
        self.assertEqual(len(bundle.manifest), 3)
        self.assertEqual(bundle.manifest[-1]["source_rows"],
                         bundle.manifest[-1]["accepted_rows"] + 1)
        self.assertEqual(bundle.observations[0].proxy_available_at.date(),
                         date(2020, 11, 4))

    def test_duplicate_and_nonnumeric_registered_date_fail(self):
        same = [("01/02/2025", "1.00"), ("01/02/2025", "1.10")]
        with self.assertRaisesRegex(TreasurySourceError, "duplicate"):
            _parse_source(_csv(same), 1)
        with self.assertRaisesRegex(TreasurySourceError, "nonnumeric"):
            _parse_source(_csv([("01/02/2025", "N/A")]), 1)

    def test_schema_change_fails(self):
        altered = _csv([("01/02/2025", "1.00")]).replace(b"10 YR", b"10-year")
        with self.assertRaisesRegex(TreasurySourceError, "columns changed"):
            _parse_source(altered, 1)


if __name__ == "__main__":
    unittest.main()
