"""Synthetic versioned OBM archive checks; no real values are read."""

from datetime import timedelta
from hashlib import sha256
from io import BytesIO
import tarfile
import unittest
from unittest.mock import patch

from research import old_coin_spend_source as source


def _archive(*, corrupt_date=False, corrupt_value=False):
    lines = [",".join(source.HEADER)]
    for i in range(source.SOURCE_ROWS):
        stamp = source.SOURCE_FIRST + timedelta(days=i)
        if corrupt_date and i == 100:
            stamp += timedelta(days=1)
        value = "-1" if corrupt_value and i == 100 else str(10000 + i)
        lines.append(f"{stamp},{source.SERIES_ID},{value},BTC,daily,OBM v0.1.0")
    member = ("\n".join(lines) + "\n").encode()
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo(source.MEMBER)
        info.size = len(member)
        archive.addfile(info, BytesIO(member))
    return stream.getvalue(), len(member)


def _parse_synthetic(body, member_bytes):
    with patch.object(source, "ARCHIVE_BYTES", len(body)), \
         patch.object(source, "ARCHIVE_SHA256", sha256(body).hexdigest()), \
         patch.object(source, "MEMBER_BYTES", member_bytes):
        return source._parse(body, source.URL)


class OldCoinSourceTests(unittest.TestCase):
    def test_pinned_schema_coverage_and_proxy(self):
        body, member_bytes = _archive()
        bundle = _parse_synthetic(body, member_bytes)
        self.assertEqual(len(bundle.observations), source.ACCEPTED_ROWS)
        self.assertEqual(bundle.observations[0].observed_date, source.FIRST)
        self.assertEqual(bundle.observations[-1].observed_date, source.LAST)
        self.assertEqual(bundle.observations[0].proxy_available_at.date(),
                         source.FIRST + timedelta(days=4))
        self.assertEqual(bundle.manifest["source_rows"], 6388)

    def test_gap_or_negative_value_rejects_whole_archive(self):
        body, size = _archive(corrupt_date=True)
        with self.assertRaisesRegex(source.OldCoinSourceError, "coverage"):
            _parse_synthetic(body, size)
        body, size = _archive(corrupt_value=True)
        with self.assertRaisesRegex(source.OldCoinSourceError, "value"):
            _parse_synthetic(body, size)

    def test_hash_or_untrusted_url_rejects(self):
        body, size = _archive()
        with self.assertRaisesRegex(source.OldCoinSourceError, "SHA-256"):
            source._parse(body, source.URL)
        with patch.object(source, "ARCHIVE_BYTES", len(body)), \
             patch.object(source, "ARCHIVE_SHA256", sha256(body).hexdigest()), \
             patch.object(source, "MEMBER_BYTES", size):
            with self.assertRaisesRegex(source.OldCoinSourceError, "untrusted"):
                source._parse(body, "https://example.com/a.tar.gz")


if __name__ == "__main__":
    unittest.main()
