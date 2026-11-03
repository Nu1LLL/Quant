"""Synthetic-only monthly Binance Vision 1m/5m ZIP integrity tests."""

from calendar import monthrange
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import json
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from research.crypto_blackbox.binance_vision_5m import (
    ArchiveIntegrityError,
    ArchiveRequest,
    archive_url,
    audit_monthly_klines,
    audit_months,
)


HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore\n"
)


def _month_rows(request: ArchiveRequest) -> list[str]:
    year, month = map(int, request.month.split("-"))
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    unit = 1_000_000 if request.market == "spot" and request.month >= "2025-01" else 1_000
    step_minutes = 1 if request.interval == "1m" else 5
    rows = []
    for offset in range(monthrange(year, month)[1] * 24 * 60 // step_minutes):
        opened = int((start + timedelta(minutes=step_minutes * offset)).timestamp()) * unit
        closed = opened + step_minutes * 60 * unit - 1
        rows.append(f"{opened},100,101,99,100,10,{closed},1000,2,6,600,0\n")
    return rows


def _source(request: ArchiveRequest, text: str, *, zip_bytes: bytes | None = None,
            checksum_override: bytes | None = None):
    url, basename = archive_url(request)
    if zip_bytes is None:
        buffer = BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(basename[:-4] + ".csv", text)
        zip_bytes = buffer.getvalue()
    checksum = (f"{sha256(zip_bytes).hexdigest()}  {basename}\n").encode("ascii")
    mapping = {url: zip_bytes, url + ".CHECKSUM": checksum_override or checksum}

    def fetch(url: str):
        payload = mapping[url]
        for index in range(0, len(payload), 97):
            yield payload[index:index + 97]

    return fetch


class BinanceVisionIntradayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = ArchiveRequest("spot", "BTCUSDT", "2024-12")
        cls.new = ArchiveRequest("spot", "ETHUSDT", "2025-01")
        cls.um = ArchiveRequest("um", "BTCUSDT", "2025-01")
        cls.old_rows = _month_rows(cls.old)
        cls.new_rows = _month_rows(cls.new)
        cls.um_rows = _month_rows(cls.um)

    def test_complete_months_across_spot_timestamp_change_and_futures(self):
        requests = (self.old, self.new, self.um)
        sources = {}
        for request, rows in zip(requests, (self.old_rows, self.new_rows, self.um_rows)):
            fetch = _source(request, HEADER + "".join(rows))
            url, _ = archive_url(request)
            sources[url] = b"".join(fetch(url))
            sources[url + ".CHECKSUM"] = b"".join(fetch(url + ".CHECKSUM"))

        def fetch(url):
            data = sources[url]
            for index in range(0, len(data), 101):
                yield data[index:index + 101]

        manifests = tuple(audit_months(requests, fetch_chunks=fetch))
        self.assertEqual([item["rows"] for item in manifests], [8928, 8928, 8928])
        self.assertEqual([item["timestamp_unit"] for item in manifests],
                         ["milliseconds", "microseconds", "milliseconds"])
        self.assertEqual(manifests[0]["last_open_utc"], "2024-12-31T23:55:00+00:00")
        self.assertFalse(any(item["publication_time_verified"] for item in manifests))
        self.assertNotIn("close", manifests[0])
        json.dumps(manifests, allow_nan=False)

    def test_complete_one_minute_month_uses_distinct_archive_and_grid(self):
        request = ArchiveRequest("spot", "BTCUSDT", "2021-02", "1m")
        url, basename = archive_url(request)
        self.assertIn("/BTCUSDT/1m/", url)
        self.assertEqual(basename, "BTCUSDT-1m-2021-02.zip")
        manifest = audit_monthly_klines(request, fetch_chunks=_source(
            request, HEADER + "".join(_month_rows(request)),
        ))
        self.assertEqual(manifest["rows"], 40_320)
        self.assertEqual(manifest["interval"], "1m")
        self.assertEqual(manifest["last_open_utc"], "2021-02-28T23:59:00+00:00")

    def test_one_minute_spot_microseconds_and_futures_milliseconds(self):
        for market, expected_unit in (("spot", "microseconds"), ("um", "milliseconds")):
            with self.subTest(market=market):
                request = ArchiveRequest(market, "BTCUSDT", "2025-01", "1m")
                manifest = audit_monthly_klines(request, fetch_chunks=_source(
                    request, HEADER + "".join(_month_rows(request)),
                ))
                self.assertEqual(manifest["rows"], 44_640)
                self.assertEqual(manifest["timestamp_unit"], expected_unit)

    def test_missing_one_minute_bar_rejected(self):
        request = ArchiveRequest("spot", "BTCUSDT", "2021-02", "1m")
        rows = _month_rows(request)
        with self.assertRaisesRegex(ArchiveIntegrityError, "missing, duplicate"):
            audit_monthly_klines(request, fetch_chunks=_source(
                request, HEADER + "".join(rows[:100] + rows[101:]),
            ))

    def test_unsupported_interval_rejected_before_fetch(self):
        with self.assertRaisesRegex(ValueError, "interval"):
            ArchiveRequest("spot", "BTCUSDT", "2021-02", "15m")

    def test_missing_five_minute_bar_rejected(self):
        rows = self.old_rows[:100] + self.old_rows[101:]
        with self.assertRaisesRegex(ArchiveIntegrityError, "missing, duplicate"):
            audit_monthly_klines(self.old,
                                 fetch_chunks=_source(self.old, HEADER + "".join(rows)))

    def test_repeated_header_is_rejected(self):
        with self.assertRaisesRegex(ArchiveIntegrityError, "repeated CSV header"):
            audit_monthly_klines(self.old, fetch_chunks=_source(
                self.old, HEADER + HEADER + "".join(self.old_rows),
            ))

    def test_wrong_timestamp_unit_and_wrong_month_rejected(self):
        rows = self.new_rows.copy()
        first = rows[0].split(",")
        first[0] = str(int(first[0]) // 1000)
        rows[0] = ",".join(first)
        with self.assertRaisesRegex(ArchiveIntegrityError, "timestamp unit"):
            audit_monthly_klines(self.new,
                                 fetch_chunks=_source(self.new, HEADER + "".join(rows)))

        rows = self.old_rows.copy()
        first = rows[0].split(",")
        first[0] = str(int(first[0]) - 300_000)
        rows[0] = ",".join(first)
        with self.assertRaisesRegex(ArchiveIntegrityError, "off-grid"):
            audit_monthly_klines(self.old,
                                 fetch_chunks=_source(self.old, HEADER + "".join(rows)))

    def test_bad_zip_and_bad_checksum_rejected(self):
        text = HEADER + "".join(self.um_rows)
        with self.assertRaisesRegex(ArchiveIntegrityError, "invalid ZIP"):
            audit_monthly_klines(self.um,
                                 fetch_chunks=_source(self.um, text, zip_bytes=b"not a zip"))
        _, basename = archive_url(self.um)
        buffer = BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(basename[:-4] + ".csv", text)
            archive.writestr("extra.csv", "not allowed")
        with self.assertRaisesRegex(ArchiveIntegrityError, "ZIP directory"):
            audit_monthly_klines(self.um, fetch_chunks=_source(
                self.um, text, zip_bytes=buffer.getvalue(),
            ))
        bad_hash = ("0" * 64 + "  BTCUSDT-5m-2025-01.zip\n").encode("ascii")
        with self.assertRaisesRegex(ArchiveIntegrityError, "SHA-256"):
            audit_monthly_klines(self.um, fetch_chunks=_source(
                self.um, text, checksum_override=bad_hash,
            ))

    def test_price_volume_and_column_defects_rejected(self):
        for column, replacement, message in (
            (1, "0", "open"),
            (5, "-1", "base volume"),
            (10, "1001", "taker buy volume"),
        ):
            with self.subTest(column=column):
                rows = self.um_rows.copy()
                first = rows[0].rstrip("\n").split(",")
                first[column] = replacement
                rows[0] = ",".join(first) + "\n"
                with self.assertRaisesRegex(ArchiveIntegrityError, message):
                    audit_monthly_klines(self.um, fetch_chunks=_source(
                        self.um, HEADER + "".join(rows),
                    ))

        rows = self.um_rows.copy()
        rows[0] = rows[0].rsplit(",", 1)[0] + "\n"
        with self.assertRaisesRegex(ArchiveIntegrityError, "12 columns"):
            audit_monthly_klines(self.um, fetch_chunks=_source(
                self.um, HEADER + "".join(rows),
            ))

    def test_zip_byte_cap_missing_source_and_duplicate_request(self):
        fetch = _source(self.old, HEADER + "".join(self.old_rows))
        with self.assertRaisesRegex(ArchiveIntegrityError, "compressed byte limit"):
            audit_monthly_klines(self.old, fetch_chunks=fetch, max_zip_bytes=64)

        def missing(_url):
            raise KeyError("missing archive")

        with self.assertRaisesRegex(ArchiveIntegrityError, "source unavailable"):
            audit_monthly_klines(self.old, fetch_chunks=missing)

        def oversized(_url):
            yield b"x" * (64 * 1024 + 1)

        with self.assertRaisesRegex(ArchiveIntegrityError, "chunk exceeds"):
            audit_monthly_klines(self.old, fetch_chunks=oversized)

        stream = audit_months((self.old, self.old), fetch_chunks=fetch)
        next(stream)
        with self.assertRaisesRegex(ValueError, "duplicate monthly"):
            next(stream)


if __name__ == "__main__":
    unittest.main()
