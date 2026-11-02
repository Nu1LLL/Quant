"""Offline fixture tests for the preregistered BTC difficulty price source."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from unittest import TestCase
from unittest.mock import patch
from zipfile import ZipFile

import requests

from research import btc_difficulty_spot as spot


UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _timestamp(stamp: datetime, unit: str) -> str:
    microseconds = int((stamp - EPOCH).total_seconds() * 1_000_000)
    return str(microseconds // 1000 if unit == "ms" else microseconds)


def _row(stamp: datetime, *, unit: str | None = None, changes: dict[int, str] | None = None) -> str:
    unit = unit or ("ms" if stamp < datetime(2025, 1, 1, tzinfo=UTC) else "us")
    tick = timedelta(milliseconds=1) if unit == "ms" else timedelta(microseconds=1)
    fields = [
        _timestamp(stamp, unit), "10", "11", "9", "10.5", "8",
        _timestamp(stamp + timedelta(days=1) - tick, unit), "80", "12", "4", "40", "0",
    ]
    for index, value in (changes or {}).items():
        fields[index] = value
    return ",".join(fields)


def _archive(month: str, text: str, *, member: str | None = None,
             extra_member: bool = False, checksum_override: str | None = None):
    url, basename = spot.archive_url(month)
    zipped = BytesIO()
    with ZipFile(zipped, "w") as handle:
        handle.writestr(member or basename[:-4] + ".csv", text)
        if extra_member:
            handle.writestr("extra.txt", "x")
    payload = zipped.getvalue()
    digest = checksum_override or sha256(payload).hexdigest()
    checksum = f"{digest}  {basename}\n".encode("ascii")

    def fetcher(requested: str) -> bytes:
        if requested == url + ".CHECKSUM":
            return checksum
        if requested == url:
            return payload
        raise AssertionError(f"unexpected URL: {requested}")

    return fetcher, payload


def _full_grid_fetcher(*, missing: datetime | None = None):
    cache: dict[str, tuple[bytes, bytes]] = {}

    def fetcher(url: str) -> bytes:
        is_checksum = url.endswith(".CHECKSUM")
        zip_url = url.removesuffix(".CHECKSUM")
        month = zip_url[-11:-4]
        if month not in cache:
            start, end = spot._month_bounds(month)
            cursor = start
            rows = []
            while cursor < end:
                if cursor != missing:
                    rows.append(_row(cursor))
                cursor += timedelta(days=1)
            monthly_fetcher, payload = _archive(month, "\n".join(rows) + "\n")
            cache[month] = (monthly_fetcher(zip_url + ".CHECKSUM"), payload)
        checksum, payload = cache[month]
        return checksum if is_checksum else payload

    return fetcher


class _Response:
    def __init__(self, status_code: int, chunks: tuple[bytes, ...] = ()):
        self.status_code = status_code
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, *, chunk_size: int):
        assert chunk_size > 0
        yield from self.chunks


class BTCDifficultySpotTests(TestCase):
    def test_verified_month_and_manifest(self):
        opened = datetime(2024, 12, 1, tzinfo=UTC)
        fetcher, payload = _archive("2024-12", _row(opened) + "\n")
        archive = spot.fetch_and_parse_month("2024-12", fetcher=fetcher)
        self.assertEqual(archive.bars[0].date.isoformat(), "2024-12-01")
        self.assertEqual(archive.bars[0].quote_volume, 80.0)
        self.assertEqual(archive.manifest.sha256, sha256(payload).hexdigest())
        self.assertEqual(archive.manifest.zip_bytes, len(payload))
        self.assertEqual(archive.manifest.rows, 1)
        self.assertEqual(
            archive.manifest.url,
            "https://data.binance.vision/data/spot/monthly/klines/"
            "BTCUSDT/1d/BTCUSDT-1d-2024-12.zip",
        )

    def test_complete_frozen_grid_and_october_endpoint(self):
        result = spot.fetch_bars(fetcher=_full_grid_fetcher())
        self.assertEqual(result.bars[0].open_time, spot.START)
        self.assertEqual(result.bars[-1].open_time, spot.LAST_OPEN)
        self.assertEqual(len(result.bars), (spot.END_EXCLUSIVE - spot.START).days)
        self.assertEqual(result.manifest[0].month, "2017-12")
        self.assertEqual(result.manifest[-1].month, "2026-10")
        self.assertEqual(result.manifest[-1].rows, 31)
        self.assertEqual(len(result.manifest), len(spot._months()))

    def test_internal_gap_is_fatal(self):
        missing = datetime(2019, 5, 12, tzinfo=UTC)
        with self.assertRaisesRegex(spot.ArchiveIntegrityError, "grid gap"):
            spot.fetch_bars(fetcher=_full_grid_fetcher(missing=missing))

    def test_checksum_and_zip_structure_fail_closed(self):
        opened = datetime(2024, 12, 1, tzinfo=UTC)
        text = _row(opened) + "\n"
        bad_cases = (
            (_archive("2024-12", text, checksum_override="0" * 64), "SHA-256 mismatch"),
            (_archive("2024-12", text, extra_member=True), "exactly"),
            (_archive("2024-12", text, member="wrong.csv"), "exactly"),
        )
        for (fetcher, _), message in bad_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(spot.ArchiveIntegrityError, message):
                    spot.fetch_and_parse_month("2024-12", fetcher=fetcher)
        with patch.object(spot, "MAX_CSV_BYTES", 10):
            fetcher, _ = _archive("2024-12", text)
            with self.assertRaisesRegex(spot.ArchiveIntegrityError, "unsafe ZIP member"):
                spot.fetch_and_parse_month("2024-12", fetcher=fetcher)

    def test_checksum_filename_and_size_are_fatal(self):
        opened = datetime(2024, 12, 1, tzinfo=UTC)
        fetcher, payload = _archive("2024-12", _row(opened) + "\n")
        url, _ = spot.archive_url("2024-12")
        def wrong_name(requested):
            if requested.endswith(".CHECKSUM"):
                return f"{sha256(payload).hexdigest()}  wrong.zip\n".encode()
            return fetcher(requested)
        with self.assertRaisesRegex(spot.ArchiveIntegrityError, "invalid CHECKSUM"):
            spot.fetch_and_parse_month("2024-12", fetcher=wrong_name)
        def oversized(requested):
            return b"x" * 257 if requested == url + ".CHECKSUM" else fetcher(requested)
        with self.assertRaisesRegex(spot.ArchiveIntegrityError, "oversized CHECKSUM"):
            spot.fetch_and_parse_month("2024-12", fetcher=oversized)

    def test_timestamp_switch_and_daily_grid(self):
        december = datetime(2024, 12, 1, tzinfo=UTC)
        january = datetime(2025, 1, 1, tzinfo=UTC)
        for month, stamp, unit in (("2024-12", december, "ms"),
                                   ("2025-01", january, "us")):
            with self.subTest(month=month):
                fetcher, _ = _archive(month, _row(stamp, unit=unit) + "\n")
                self.assertEqual(spot.fetch_and_parse_month(month, fetcher=fetcher)
                                 .bars[0].open_time, stamp)
        bad_rows = (
            ("2024-12", _row(december, unit="us")),
            ("2025-01", _row(january, unit="ms")),
            ("2024-12", _row(december + timedelta(hours=1))),
            ("2024-12", _row(december, changes={6: _timestamp(
                december + timedelta(days=1) - timedelta(seconds=1), "ms")})),
            ("2024-12", _row(december, changes={6: _timestamp(
                december + timedelta(days=1) - timedelta(microseconds=1), "us")})),
        )
        for month, row in bad_rows:
            with self.subTest(month=month, row=row):
                fetcher, _ = _archive(month, row + "\n")
                with self.assertRaises(spot.ArchiveIntegrityError):
                    spot.fetch_and_parse_month(month, fetcher=fetcher)

    def test_schema_ohlc_volume_and_duplicates(self):
        stamp = datetime(2024, 12, 1, tzinfo=UTC)
        good = _row(stamp)
        bad_rows = (
            good + ",extra",                    # 13 fields
            ",".join(good.split(",")[:-1]),  # 11 fields
            _row(stamp, changes={2: "9"}),     # high below open/close
            _row(stamp, changes={3: "11"}),    # low above open/close
            _row(stamp, changes={5: "-1"}),    # base volume
            _row(stamp, changes={7: "-1"}),    # quote volume
            _row(stamp, changes={7: "0"}),     # base/quote mismatch
            _row(stamp, changes={8: "1.5"}),   # noninteger count
            _row(stamp, changes={9: "nan"}),   # nonfinite taker volume
        )
        for row in bad_rows:
            with self.subTest(row=row):
                fetcher, _ = _archive("2024-12", row + "\n")
                with self.assertRaises(spot.ArchiveIntegrityError):
                    spot.fetch_and_parse_month("2024-12", fetcher=fetcher)
        for text in (good + "\n" + good + "\n",
                     _row(stamp + timedelta(days=1)) + "\n" + good + "\n"):
            fetcher, _ = _archive("2024-12", text)
            with self.assertRaisesRegex(spot.ArchiveIntegrityError, "duplicate or unsorted"):
                spot.fetch_and_parse_month("2024-12", fetcher=fetcher)

    def test_http_retry_cap_and_missing_archive(self):
        with patch.object(spot.requests, "get", side_effect=[
            _Response(503), _Response(200, (b"verified",))
        ]) as get, patch.object(spot.time, "sleep") as sleep:
            self.assertEqual(spot._http_fetch("https://example.test/archive.zip"), b"verified")
            self.assertEqual(get.call_count, 2)
            sleep.assert_called_once()
        with patch.object(spot.requests, "get", side_effect=[
            requests.exceptions.ChunkedEncodingError("truncated"),
            _Response(200, (b"verified",))
        ]), patch.object(spot.time, "sleep"):
            self.assertEqual(spot._http_fetch("https://example.test/archive.zip"), b"verified")
        with patch.object(spot.requests, "get", return_value=_Response(404)):
            with self.assertRaises(spot.ArchiveUnavailable):
                spot._http_fetch("https://example.test/archive.zip")
        with patch.object(spot, "MAX_ZIP_BYTES", 3), patch.object(
            spot.requests, "get", return_value=_Response(200, (b"1234",))
        ):
            with self.assertRaisesRegex(spot.ArchiveIntegrityError, "download exceeds"):
                spot._http_fetch("https://example.test/archive.zip")
