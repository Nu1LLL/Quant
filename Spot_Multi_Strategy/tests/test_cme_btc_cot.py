"""Synthetic-only tests for the frozen CME BTC CFTC TFF signal."""

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import csv
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.cme_btc_cot import (
    BLACKOUTS,
    COTDataError,
    COTReport,
    YEARS,
    archive_url,
    desired_exposure,
    fetch_cot,
    parse_annual_zip,
)


UTC = timezone.utc
HEADER = (
    "Market_and_Exchange_Names", "Report_Date_as_YYYY-MM-DD",
    "CFTC_Contract_Market_Code", "Open_Interest_All",
    "Asset_Mgr_Positions_Long_All", "Asset_Mgr_Positions_Short_All",
    "FutOnly_or_Combined",
)
MARKET = "BITCOIN - CHICAGO MERCANTILE EXCHANGE"


def _row(day: date, *, long="40", short="20", oi="100", code="133741",
         market=MARKET, kind="FutOnly") -> tuple[str, ...]:
    return market, day.isoformat(), code, oi, long, short, kind


def _zip(year: int, rows: list[tuple[str, ...]], *, header=HEADER,
         member=None, extra_member=False) -> bytes:
    text = StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(header)
    writer.writerows(rows)
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as zipped:
        zipped.writestr(member or f"FinFut{year % 100:02d}.txt", text.getvalue())
        if extra_member:
            zipped.writestr("readme.txt", "extra")
    return output.getvalue()


def _weekly_year(year: int) -> list[tuple[str, ...]]:
    day = (date(2018, 4, 10) if year == 2018 else
           date(2019, 1, 8) if year == 2019 else date(year, 1, 1))
    while day.weekday() != 1:
        day += timedelta(days=1)
    rows = []
    while day.year == year:
        rows.append(_row(day, long=str(20 + len(rows))))
        day += timedelta(days=7)
    if year == 2018:
        rows.append(_row(date(2018, 12, 31), long=str(20 + len(rows))))
        before_listing = []
        day = date(2018, 1, 2)
        while day < date(2018, 4, 10):
            before_listing.append(_row(day, code="000000", market="OTHER CME MARKET"))
            day += timedelta(days=7)
        rows = before_listing + rows
    return rows


def _reports(start: date, nets: list[int], *, oi=100,
             step=7) -> tuple[COTReport, ...]:
    return tuple(COTReport(start + timedelta(days=i * step),
                           long=50 + net, short=50, open_interest=oi)
                 for i, net in enumerate(nets))


class CmeBtcCotTests(unittest.TestCase):
    def test_official_url_and_bounded_annual_parser_manifest(self):
        self.assertEqual(
            archive_url(2025),
            "https://www.cftc.gov/files/dea/history/fut_fin_txt_2025.zip",
        )
        with self.assertRaises(ValueError):
            archive_url(2026)
        rows = [_row(date(2024, 1, 2)),
                _row(date(2024, 1, 9), code="133742", market="MICRO BITCOIN")]
        payload = _zip(2024, rows)
        parsed = parse_annual_zip(payload, 2024)
        self.assertEqual(parsed.reports, (COTReport(date(2024, 1, 2), 40, 20, 100),))
        self.assertEqual(parsed.manifest["sha256"], sha256(payload).hexdigest())
        self.assertEqual(parsed.manifest["zip_bytes"], len(payload))
        self.assertEqual(parsed.manifest["member"], "FinFut24.txt")
        self.assertEqual(parsed.manifest["header"], list(HEADER))
        self.assertEqual(parsed.manifest["csv_rows"], 2)
        self.assertEqual(parsed.manifest["report_rows"], 1)
        self.assertEqual(parsed.manifest["first_report_date"], "2024-01-02")
        json.dumps(parsed.manifest, allow_nan=False)

    def test_wrong_schema_contract_exchange_flag_and_date_fail(self):
        day = date(2024, 1, 2)
        cases = (
            (_zip(2024, [_row(day)], header=HEADER[:-1]), "missing required"),
            (_zip(2024, [_row(day)], header=HEADER[:-1] + ("Wrong",)), "missing required"),
            (_zip(2024, [_row(day, market="BITCOIN - OTHER EXCHANGE")]), "wrong BTC market"),
            (_zip(2024, [_row(day, kind="Combined")]), "not TFF futures only"),
            (_zip(2024, [_row(day, code="133742")]), "rows absent"),
            (_zip(2024, [_row(date(2023, 12, 26))]), "outside archive year"),
        )
        for payload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(COTDataError, message):
                parse_annual_zip(payload, 2024)

    def test_invalid_positions_duplicate_rows_and_missing_weekly_market_fail(self):
        day = date(2024, 1, 2)
        cases = (
            ([_row(day, long="nan")], "nonnegative finite integer"),
            ([_row(day, short="-1")], "nonnegative finite integer"),
            ([_row(day, long="4.0")], "nonnegative finite integer"),
            ([_row(day, oi="0")], "zero open interest"),
            ([_row(day, long="101")], "position exceeds open interest"),
            ([_row(day), _row(day)], "duplicate"),
            ([_row(day), _row(day + timedelta(days=7), code="133742"),
              _row(day + timedelta(days=14))], "row absent"),
        )
        for rows, message in cases:
            with self.subTest(rows=rows), self.assertRaisesRegex(COTDataError, message):
                parse_annual_zip(_zip(2024, rows), 2024)

    def test_unsafe_or_corrupt_zip_fails_before_any_signal(self):
        day = date(2024, 1, 2)
        cases = (
            b"not a ZIP",
            _zip(2024, [_row(day)], extra_member=True),
            _zip(2024, [_row(day)], member="../FinFut24.txt"),
            _zip(2024, [_row(day)], member="FinFut24.bin"),
        )
        for payload in cases:
            with self.subTest(payload_size=len(payload)), self.assertRaises(COTDataError):
                parse_annual_zip(payload, 2024)
        payload = _zip(2024, [_row(day)])
        with patch("research.cme_btc_cot.MAX_ZIP_BYTES", len(payload) - 1):
            with self.assertRaisesRegex(COTDataError, "oversized"):
                parse_annual_zip(payload, 2024)

    def test_fetch_cot_requires_all_eight_complete_years(self):
        source = {archive_url(year): _zip(year, _weekly_year(year)) for year in YEARS}
        dataset = fetch_cot(fetcher=source.__getitem__)
        self.assertEqual(dataset.manifest["archive_count"], 8)
        self.assertEqual(tuple(item["year"] for item in dataset.manifest["archives"]), YEARS)
        self.assertEqual(dataset.manifest["report_rows"], len(dataset.reports))
        self.assertEqual(dataset.manifest["anomalies"], [])
        json.dumps(dataset.manifest, allow_nan=False)
        del source[archive_url(2022)]
        with self.assertRaisesRegex(COTDataError, "archive unavailable"):
            fetch_cot(fetcher=source.__getitem__)
        source[archive_url(2022)] = _zip(2022, [_row(date(2022, 1, 4))])
        with self.assertRaisesRegex(COTDataError, "too few weekly"):
            fetch_cot(fetcher=source.__getitem__)

    def test_signal_d_plus_fourteen_and_twenty_one_day_age(self):
        reports = _reports(date(2024, 1, 2), [0, 1, 2, 3, 4])
        latest = reports[-1].report_date
        self.assertFalse(desired_exposure(
            reports, datetime.combine(latest + timedelta(days=13),
                                      datetime.min.time(), UTC)))
        self.assertTrue(desired_exposure(
            reports, datetime.combine(latest + timedelta(days=14),
                                      datetime.min.time(), UTC)))
        self.assertTrue(desired_exposure(
            reports, datetime.combine(latest + timedelta(days=21),
                                      datetime.min.time(), UTC)))
        self.assertFalse(desired_exposure(
            reports, datetime.combine(latest + timedelta(days=22),
                                      datetime.min.time(), UTC)))

    def test_signal_uses_normalized_net_and_strict_positive_delta(self):
        reports = list(_reports(date(2024, 1, 2), [0, 1, 2, 3, 4]))
        when = datetime(2024, 2, 13, tzinfo=UTC)
        self.assertTrue(desired_exposure(reports, when))
        # Raw net rises, but net/open-interest ratio falls.
        reports[0] = COTReport(reports[0].report_date, 60, 50, 100)
        reports[-1] = COTReport(reports[-1].report_date, 70, 50, 400)
        self.assertFalse(desired_exposure(reports, when))
        reports[0] = COTReport(reports[0].report_date, 60, 50, 100)
        reports[-1] = COTReport(reports[-1].report_date, 70, 50, 200)
        self.assertFalse(desired_exposure(reports, when))

    def test_blackouts_exclude_reports_and_break_consecutive_history(self):
        start, end = BLACKOUTS[1]
        self.assertEqual((start, end), (date(2023, 1, 31), date(2023, 3, 14)))
        reports = _reports(date(2023, 1, 3), list(range(16)))
        self.assertFalse(desired_exposure(reports, datetime(2023, 3, 29, tzinfo=UTC)))
        self.assertFalse(desired_exposure(reports, datetime(2023, 4, 11, tzinfo=UTC)))
        self.assertTrue(desired_exposure(reports, datetime(2023, 5, 2, tzinfo=UTC)))

    def test_signal_requires_four_adjacent_prior_reports_and_utc_midnight(self):
        reports = _reports(date(2024, 1, 2), [0, 1, 2, 3, 4])
        self.assertFalse(desired_exposure(reports[:-1], datetime(2024, 2, 13, tzinfo=UTC)))
        gap = reports[:2] + tuple(COTReport(report.report_date + timedelta(days=7),
                                          report.long, report.short, report.open_interest)
                                  for report in reports[2:])
        self.assertFalse(desired_exposure(gap, datetime(2024, 2, 20, tzinfo=UTC)))
        with self.assertRaisesRegex(COTDataError, "duplicate"):
            desired_exposure(reports + (reports[-1],), datetime(2024, 2, 13, tzinfo=UTC))
        with self.assertRaises(ValueError):
            desired_exposure(reports, datetime(2024, 2, 13))
        with self.assertRaises(ValueError):
            desired_exposure(reports, datetime(2024, 2, 13, 1, tzinfo=UTC))


if __name__ == "__main__":
    unittest.main()
