"""Synthetic, offline tests for the frozen CME BTC spot archive loader."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from unittest import TestCase
from unittest.mock import patch
from zipfile import ZipFile

import requests

from research import btc_difficulty_spot as shared_archive
from research import cme_btc_spot as spot


UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _timestamp(stamp: datetime, unit: str) -> str:
    delta = stamp - EPOCH
    microseconds = ((delta.days * 86400 + delta.seconds) * 1_000_000
                    + delta.microseconds)
    return str(microseconds // 1000 if unit == "ms" else microseconds)


def _row(stamp: datetime, *, unit: str | None = None,
         changes: dict[int, str] | None = None) -> str:
    unit = unit or ("ms" if stamp.year < 2025 else "us")
    tick = timedelta(milliseconds=1) if unit == "ms" else timedelta(microseconds=1)
    fields = [
        _timestamp(stamp, unit), "100", "110", "90", "101", "3",
        _timestamp(stamp + timedelta(days=1) - tick, unit),
        "300", "10", "1", "100", "0",
    ]
    for index, value in (changes or {}).items():
        fields[index] = value
    return ",".join(fields)


def _monthly_rows(month: str, *, missing: datetime | None = None,
                  changed_day: datetime | None = None,
                  changes: dict[int, str] | None = None,
                  unit: str | None = None) -> list[str]:
    start, end = spot._month_bounds(month)
    rows: list[str] = []
    cursor = start
    while cursor < end:
        if cursor != missing:
            rows.append(_row(cursor, unit=unit,
                             changes=changes if cursor == changed_day else None))
        cursor += timedelta(days=1)
    return rows


def _fixture(month: str, text: str, *, member: str | None = None,
             extra_member: bool = False,
             checksum_name: str | None = None,
             checksum_digest: str | None = None):
    url, basename = spot.archive_url(month)
    zipped = BytesIO()
    with ZipFile(zipped, "w") as handle:
        handle.writestr(member or basename.removesuffix(".zip") + ".csv", text)
        if extra_member:
            handle.writestr("extra.csv", "unexpected")
    payload = zipped.getvalue()
    digest = checksum_digest or sha256(payload).hexdigest()
    checksum = f"{digest}  {checksum_name or basename}\n".encode("ascii")

    def fetcher(requested: str) -> bytes:
        if requested == url + ".CHECKSUM":
            return checksum
        if requested == url:
            return payload
        raise AssertionError(f"unexpected URL: {requested}")

    return fetcher, payload


def _full_grid_fetcher(*, missing_month: str | None = None,
                       missing_day: datetime | None = None):
    cache: dict[str, tuple[bytes, bytes]] = {}

    def fetcher(url: str) -> bytes:
        checksum_requested = url.endswith(".CHECKSUM")
        zip_url = url.removesuffix(".CHECKSUM")
        month = zip_url[-11:-4]
        if month not in cache:
            rows = _monthly_rows(
                month, missing=missing_day if month == missing_month else None
            )
            monthly_fetcher, payload = _fixture(month, "\n".join(rows) + "\n")
            cache[month] = (monthly_fetcher(zip_url + ".CHECKSUM"), payload)
        checksum, payload = cache[month]
        return checksum if checksum_requested else payload

    return fetcher


class CMEBTCSpotTests(TestCase):
    def test_frozen_months_full_grid_and_endpoint(self):
        self.assertEqual(spot._months()[0], "2018-12")
        self.assertEqual(spot._months()[-1], "2026-01")
        self.assertEqual(len(spot._months()), 86)
        result = spot.fetch_bars(fetcher=_full_grid_fetcher())
        self.assertEqual(result.bars[0].open_time, spot.START)
        self.assertEqual(result.bars[-1].open_time, spot.LAST_OPEN)
        self.assertEqual(len(result.bars), (spot.END_EXCLUSIVE - spot.START).days)
        self.assertEqual(len(result.manifest), 86)
        self.assertEqual(result.manifest[-1].month, "2026-01")
        self.assertEqual(result.manifest[-1].rows, 31)
        self.assertEqual(result.bars[-1].quote_volume, 300.0)
        self.assertEqual(result.bars[-1].open, 100.0)

    def test_verified_month_manifest_and_exact_url(self):
        month = "2024-12"
        fetcher, payload = _fixture(month, "\n".join(_monthly_rows(month)) + "\n")
        result = spot.fetch_and_parse_month(month, fetcher=fetcher)
        self.assertEqual(len(result.bars), 31)
        self.assertEqual(result.bars[0].date.isoformat(), "2024-12-01")
        self.assertEqual(result.manifest.sha256, sha256(payload).hexdigest())
        self.assertEqual(result.manifest.zip_bytes, len(payload))
        self.assertEqual(result.manifest.rows, 31)
        self.assertEqual(
            result.manifest.url,
            "https://data.binance.vision/data/spot/monthly/klines/"
            "BTCUSDT/1d/BTCUSDT-1d-2024-12.zip",
        )

    def test_month_gap_and_endpoint_month_gap_fail_closed(self):
        for month, missing in (
            ("2019-05", datetime(2019, 5, 12, tzinfo=UTC)),
            ("2026-01", datetime(2026, 1, 2, tzinfo=UTC)),
        ):
            with self.subTest(month=month):
                rows = _monthly_rows(month, missing=missing)
                fetcher, _ = _fixture(month, "\n".join(rows) + "\n")
                with self.assertRaisesRegex(spot.ArchiveIntegrityError, "grid"):
                    spot.fetch_and_parse_month(month, fetcher=fetcher)

    def test_checksum_zip_member_and_size_fail_closed(self):
        month = "2024-12"
        text = "\n".join(_monthly_rows(month)) + "\n"
        cases = (
            (_fixture(month, text, checksum_digest="0" * 64), "SHA-256 mismatch"),
            (_fixture(month, text, checksum_name="wrong.zip"), "invalid CHECKSUM"),
            (_fixture(month, text, member="wrong.csv"), "exactly"),
            (_fixture(month, text, extra_member=True), "exactly"),
        )
        for (fetcher, _), message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(spot.ArchiveIntegrityError, message):
                    spot.fetch_and_parse_month(month, fetcher=fetcher)
        fetcher, _ = _fixture(month, text)
        with patch.object(shared_archive, "MAX_CSV_BYTES", 100):
            with self.assertRaisesRegex(spot.ArchiveIntegrityError, "unsafe ZIP member"):
                spot.fetch_and_parse_month(month, fetcher=fetcher)

    def test_timestamp_units_and_daily_close(self):
        for month, unit in (("2024-12", "ms"), ("2025-01", "us")):
            with self.subTest(month=month):
                rows = _monthly_rows(month, unit=unit)
                fetcher, _ = _fixture(month, "\n".join(rows) + "\n")
                self.assertEqual(spot.fetch_and_parse_month(month, fetcher=fetcher)
                                 .bars[0].open_time, spot._month_bounds(month)[0])
                wrong_unit = "us" if unit == "ms" else "ms"
                rows = _monthly_rows(month, unit=wrong_unit)
                fetcher, _ = _fixture(month, "\n".join(rows) + "\n")
                with self.assertRaises(spot.ArchiveIntegrityError):
                    spot.fetch_and_parse_month(month, fetcher=fetcher)
        month = "2024-12"
        first = spot._month_bounds(month)[0]
        early_close = _timestamp(first + timedelta(hours=23), "ms")
        rows = _monthly_rows(month, changed_day=first, changes={6: early_close})
        fetcher, _ = _fixture(month, "\n".join(rows) + "\n")
        with self.assertRaisesRegex(spot.ArchiveIntegrityError, "close_time"):
            spot.fetch_and_parse_month(month, fetcher=fetcher)

    def test_schema_ohlc_and_quote_volume_fail_closed(self):
        month = "2024-12"
        first = spot._month_bounds(month)[0]
        cases = (
            ({2: "99"}, "OHLC"),
            ({3: "102"}, "OHLC"),
            ({7: "-1"}, "quote_volume"),
            ({7: "0"}, "base/quote"),
            ({7: "nan"}, "quote_volume"),
            ({8: "1.5"}, "trade count"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                rows = _monthly_rows(month, changed_day=first, changes=changes)
                fetcher, _ = _fixture(month, "\n".join(rows) + "\n")
                with self.assertRaisesRegex(spot.ArchiveIntegrityError, message):
                    spot.fetch_and_parse_month(month, fetcher=fetcher)
        rows = _monthly_rows(month)
        for bad_text in (
            "\n".join([rows[0] + ",extra", *rows[1:]]) + "\n",
            "\n".join([",".join(rows[0].split(",")[:-1]), *rows[1:]]) + "\n",
            ",".join(["wrong_header", *shared_archive.BAR_HEADER[1:]])
            + "\n" + "\n".join(rows) + "\n",
        ):
            fetcher, _ = _fixture(month, bad_text)
            with self.assertRaises(spot.ArchiveIntegrityError):
                spot.fetch_and_parse_month(month, fetcher=fetcher)

    def test_missing_official_archive_is_data_failure(self):
        def unavailable(_url: str) -> bytes:
            raise requests.HTTPError("404")

        with self.assertRaises(spot.ArchiveUnavailable):
            spot.fetch_and_parse_month("2024-12", fetcher=unavailable)
        with self.assertRaises(ValueError):
            spot.fetch_and_parse_month("2018-11", fetcher=unavailable)
