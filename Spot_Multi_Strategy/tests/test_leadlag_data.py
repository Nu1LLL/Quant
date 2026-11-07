from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile
import unittest

from research.leadlag_data import (
    FUNDING_MANIFEST, PRICE_MANIFEST, LeadLagDataError, SYMBOLS,
    _funding_month, _pins, _price_month, _verified_zip,
)


UTC = timezone.utc


def packed(name: str, text: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr(name[:-4] + ".csv", text)
    return buffer.getvalue()


class LeadLagDataTests(unittest.TestCase):
    def test_manifests_cover_frozen_ten_coin_universe(self):
        self.assertEqual(len(SYMBOLS), 10)
        self.assertEqual(len(_pins(PRICE_MANIFEST, kind="5m")), 610)
        self.assertEqual(len(_pins(FUNDING_MANIFEST, kind="fundingRate")), 600)

    def test_zip_must_match_committed_and_official_digests(self):
        name = "ADAUSDT-5m-2020-12.zip"
        blob = packed(name, "test\n")
        digest = sha256(blob).hexdigest()
        with patch("research.leadlag_data._fetch",
                   side_effect=lambda url, **_: f"{digest}  {name}\n".encode()
                   if url.endswith(".CHECKSUM") else blob):
            with _verified_zip("ADAUSDT", "2020-12", "5m", digest) as archive:
                self.assertEqual(len(archive.infolist()), 1)
            with self.assertRaisesRegex(LeadLagDataError, "digest"):
                _verified_zip("ADAUSDT", "2020-12", "5m", "0" * 64)

    def test_price_reader_rejects_bad_ohlc_after_full_grid(self):
        name = "ADAUSDT-5m-2020-12.zip"
        first = int(datetime(2020, 12, 1, tzinfo=UTC).timestamp() * 1000)

        def body(bad: bool) -> bytes:
            lines = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"]
            for index in range(31 * 288):
                stamp = first + index * 300_000
                high = .5 if bad and index == 10 else 2
                lines.append(f"{stamp},1,{high},0.5,1,10,{stamp+299999},10,1,5,5,0")
            return packed(name, "\n".join(lines) + "\n")

        pin = {"sha256": "0" * 64, "bars": str(31 * 288)}
        with patch("research.leadlag_data._verified_zip",
                   side_effect=lambda *_: ZipFile(BytesIO(body(False)))):
            self.assertEqual(_price_month("ADAUSDT", "2020-12", pin).shape,
                             (31 * 288, 4))
        with patch("research.leadlag_data._verified_zip",
                   side_effect=lambda *_: ZipFile(BytesIO(body(True)))):
            with self.assertRaisesRegex(LeadLagDataError, "invalid price"):
                _price_month("ADAUSDT", "2020-12", pin)

    def test_funding_reader_validates_time_without_accessing_rate(self):
        name = "ADAUSDT-fundingRate-2021-01.zip"
        start = datetime(2021, 1, 1, tzinfo=UTC)
        lines = ["calc_time,funding_interval_hours,last_funding_rate"]
        for index in range(31 * 3):
            stamp = int((start + timedelta(hours=index * 8)).timestamp() * 1000)
            lines.append(f"{stamp},8,UNREAD_RATE")
        blob = packed(name, "\n".join(lines) + "\n")
        pin = {"sha256": "0" * 64, "events": "93"}
        with patch("research.leadlag_data._verified_zip",
                   side_effect=lambda *_: ZipFile(BytesIO(blob))):
            self.assertEqual(_funding_month("ADAUSDT", "2021-01", pin), 93)


if __name__ == "__main__":
    unittest.main()
