"""Synthetic official-archive transport tests; no market outcomes are read."""

from datetime import date, datetime, timezone
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from research.crypto_blackbox.binance_vision_daily_prospective import (
    ACTIVATION_DATE, ArchivePending, SourceFailure, SYMBOLS, _archive_urls,
    capture_day,
)


DAY = date(2026, 11, 13)
CAPTURE = datetime(2026, 11, 14, 4, 17, tzinfo=timezone.utc)


def _fake_fetch(*, bad_checksum=False, missing_symbol=None, id_gap=False,
                missing_hour=False):
    payloads = {}
    start_us = int(datetime(2026, 11, 13, tzinfo=timezone.utc).timestamp()) * 1_000_000
    for symbol in SYMBOLS:
        url, checksum_url, filename = _archive_urls(DAY, symbol)
        rows = []
        for hour in range(24 if not missing_hour else 23):
            agg_id = 1000 + hour + (1 if id_gap and hour >= 12 else 0)
            rows.append(f"{agg_id},1.25,2,3000,3000,{start_us + hour * 3600000000 + 1000000},False,True\n")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(filename[:-4] + ".csv", "".join(rows))
        raw = buffer.getvalue()
        digest = sha256(raw).hexdigest()
        if bad_checksum and symbol == SYMBOLS[-1]:
            digest = "0" * 64
        payloads[url] = raw
        payloads[checksum_url] = f"{digest}  {filename}\n".encode()

    def fetch(url: str, maximum: int) -> bytes:
        if missing_symbol and f"/{missing_symbol}/" in url:
            raise ArchivePending("synthetic missing publication")
        raw = payloads[url]
        if len(raw) > maximum:
            raise SourceFailure("synthetic bound exceeded")
        return raw

    return fetch


class BinanceVisionDailyTests(unittest.TestCase):
    def test_captures_all_symbols_once_without_raw_tape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = capture_day(DAY, root, fetch=_fake_fetch(),
                                 captured_at=CAPTURE, code_commit="synthetic")
            body = json.loads(gzip.decompress(target.read_bytes()))
            self.assertEqual(body["day"], DAY.isoformat())
            self.assertEqual(body["captured_at_utc"], CAPTURE.isoformat())
            self.assertEqual(len(body["sources"]), 12)
            self.assertEqual(len(body["sources"][0]["hours"]), 24)
            self.assertEqual(body["sources"][0]["hours"][0]["buy_quote_micros"], 2_500_000)
            self.assertFalse(body["independent_executable_oos"])
            self.assertLess(target.stat().st_size, 20_000)
            before = target.read_bytes()
            self.assertEqual(capture_day(DAY, root, fetch=_fake_fetch(bad_checksum=True),
                                         captured_at=CAPTURE), target)
            self.assertEqual(target.read_bytes(), before)

    def test_checksum_failure_leaves_no_partial_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(SourceFailure, "SHA-256 mismatch"):
                capture_day(DAY, root, fetch=_fake_fetch(bad_checksum=True), captured_at=CAPTURE)
            self.assertFalse(list(root.glob("*.json.gz")))

    def test_missing_archive_leaves_no_partial_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ArchivePending):
                capture_day(DAY, root, fetch=_fake_fetch(missing_symbol="BTCUSDT"),
                            captured_at=CAPTURE)
            self.assertFalse(list(root.glob("*.json.gz")))

    def test_id_gap_or_empty_hour_fails_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SourceFailure):
                capture_day(DAY, root, fetch=_fake_fetch(id_gap=True), captured_at=CAPTURE)
            with self.assertRaises(SourceFailure):
                capture_day(DAY, root, fetch=_fake_fetch(missing_hour=True), captured_at=CAPTURE)

    def test_does_not_capture_today_or_before_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                capture_day(ACTIVATION_DATE, Path(tmp), fetch=_fake_fetch(),
                            captured_at=datetime(2026, 11, 13, 23, tzinfo=timezone.utc))
            with self.assertRaises(ValueError):
                capture_day(date(2026, 11, 12), Path(tmp), fetch=_fake_fetch(),
                            captured_at=CAPTURE)


if __name__ == "__main__":
    unittest.main()
