"""Synthetic-only checks for bounded historical aggregate-trade intake."""

from hashlib import sha256
from io import BytesIO
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from research.crypto_blackbox.binance_vision_5m import ArchiveIntegrityError
from research.crypto_blackbox.binance_vision_aggtrades import (
    AggTradeRequest, archive_url, audit_month, verified_month,
    verify_adjacent_months,
)


def source(request, rows, *, wrong_hash=False):
    url, basename = archive_url(request)
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr(basename[:-4] + ".csv", "".join(rows))
    payload = output.getvalue()
    digest = "0" * 64 if wrong_hash else sha256(payload).hexdigest()
    bodies = {url: payload,
              url + ".CHECKSUM": f"{digest}  {basename}\n".encode()}

    def fetch(path):
        data = bodies[path]
        for offset in range(0, len(data), 31):
            yield data[offset:offset + 31]

    return fetch


class AggTradeArchiveTests(unittest.TestCase):
    def test_checksum_validated_rows_and_timestamp_switch(self):
        for month, stamp, expected_unit in (
            ("2024-12", "1733011200000", "milliseconds"),
            ("2025-01", "1735689600000000", "microseconds"),
        ):
            with self.subTest(month=month):
                request = AggTradeRequest("SOLUSDT", month)
                rows = [f"10,12.5,4,100,101,{stamp},False,True\n",
                        f"11,12.6,1,102,102,{int(stamp)+1},True,True\n"]
                fetch = source(request, rows)
                manifest = audit_month(request, fetch_chunks=fetch)
                self.assertEqual(manifest["rows"], 2)
                self.assertEqual(manifest["timestamp_unit"], expected_unit)
                self.assertFalse(manifest["publication_time_verified"])
                self.assertNotIn("price", manifest)
                with verified_month(request, fetch_chunks=fetch) as events:
                    first, second = tuple(events)
                self.assertEqual(first.price, 12.5)
                self.assertFalse(first.buyer_is_maker)
                self.assertTrue(second.buyer_is_maker)
                self.assertEqual(second.trade_time_us - first.trade_time_us,
                                 1000 if month == "2024-12" else 1)
                self.assertFalse(first.archive_publication_time_verified)

    def test_wrong_checksum_blocks_rows_before_context_entry(self):
        request = AggTradeRequest("DOGEUSDT", "2025-01")
        entered = False
        with self.assertRaisesRegex(ArchiveIntegrityError, "SHA-256"):
            with verified_month(request, fetch_chunks=source(request, [
                "1,0.2,10,2,3,1735689600000000,False,True\n",
            ], wrong_hash=True)):
                entered = True
        self.assertFalse(entered)

    def test_id_gap_and_wrong_timestamp_unit_fail_closed(self):
        request = AggTradeRequest("DOGEUSDT", "2025-01")
        base = "1,0.2,10,2,3,1735689600000000,False,True\n"
        with self.assertRaisesRegex(ArchiveIntegrityError, "aggTrade ID"):
            audit_month(request, fetch_chunks=source(request, [
                base, "3,0.2,10,4,5,1735689600000001,True,True\n",
            ]))
        with self.assertRaisesRegex(ArchiveIntegrityError, "timestamp unit"):
            audit_month(request, fetch_chunks=source(request, [
                "1,0.2,10,2,3,1735689600000,False,True\n",
            ]))

    def test_storage_cap_and_cross_month_gap(self):
        request = AggTradeRequest("DOGEUSDT", "2025-01")
        with self.assertRaisesRegex(ArchiveIntegrityError, "temporary-storage limit"):
            audit_month(request, fetch_chunks=source(request, [
                "1,0.2,10,2,3,1735689600000000,False,True\n",
            ]), max_zip_bytes=10)
        left = {"symbol": "DOGEUSDT", "month": "2024-12", "last_agg_id": 10,
                "last_raw_trade_id": 100, "last_trade_time_us": 1735689599000000}
        right = {"symbol": "DOGEUSDT", "month": "2025-01", "first_agg_id": 11,
                 "first_raw_trade_id": 101, "first_trade_time_us": 1735689600000000}
        verify_adjacent_months(left, right)
        right["first_agg_id"] = 12
        with self.assertRaisesRegex(ArchiveIntegrityError, "gap across months"):
            verify_adjacent_months(left, right)


if __name__ == "__main__":
    unittest.main()
