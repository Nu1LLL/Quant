from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile
import unittest

from research.funding_spread_data import (
    FUNDING_MANIFEST, KLINE_MANIFEST, FundingEvent, FundingSpreadDataError,
    HourBar, SYMBOLS, TerminalOpen, _funding_month, _pins, _price_month, _verified_zip,
    iter_hours, load_funding,
)


UTC = timezone.utc


def zip_payload(name: str, csv_body: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr(name[:-4] + ".csv", csv_body)
    return buffer.getvalue()


class FundingSpreadDataTests(unittest.TestCase):
    def test_locked_manifests_cover_complete_months(self):
        self.assertEqual(len(_pins(KLINE_MANIFEST, kind="1h")), 186)
        self.assertEqual(len(_pins(FUNDING_MANIFEST, kind="fundingRate")), 183)

    def test_archive_hash_must_match_both_pin_and_official_checksum(self):
        name = "ADAUSDT-1h-2020-12.zip"
        body = zip_payload(name, "test\n")
        digest = sha256(body).hexdigest()
        with patch("research.funding_spread_data._fetch",
                   side_effect=lambda url, **_: (digest + "  " + name + "\n").encode()
                   if url.endswith(".CHECKSUM") else body):
            with _verified_zip("ADAUSDT", "2020-12", "1h", digest) as archive:
                self.assertEqual(len(archive.infolist()), 1)
            with self.assertRaises(FundingSpreadDataError):
                _verified_zip("ADAUSDT", "2020-12", "1h", "0" * 64)

    def test_hourly_full_grid_and_ohlc_checked_before_rows_exposed(self):
        month = "2020-12"
        start = int(datetime(2020, 12, 1, tzinfo=UTC).timestamp() * 1000)
        def rows(*, bad=False):
            out = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"]
            for index in range(31 * 24):
                stamp = start + index * 3_600_000
                if bad and index == 20:
                    stamp += 3_600_000
                out.append(f"{stamp},1,2,0.5,1,100,{stamp+3_599_999},100,1,50,50,0")
            return "\n".join(out) + "\n"
        def archive_for(body):
            return ZipFile(BytesIO(zip_payload("ADAUSDT-1h-2020-12.zip", body)))
        with patch("research.funding_spread_data._verified_zip",
                   side_effect=lambda *_: archive_for(rows())):
            self.assertEqual(len(_price_month("ADAUSDT", month, "0" * 64)), 744)
        with patch("research.funding_spread_data._verified_zip",
                   side_effect=lambda *_: archive_for(rows(bad=True))):
            with self.assertRaisesRegex(FundingSpreadDataError, "missing"):
                _price_month("ADAUSDT", month, "0" * 64)

    def test_only_last_held_hour_and_exit_open_are_exposed_from_2026(self):
        first = datetime(2026, 1, 1, tzinfo=UTC)
        pins = {("2026-01", symbol): "0" * 64 for symbol in SYMBOLS}
        fake_prices = ((1.0, 1.1, .9, 1.05), (1.06, 1.2, 1.0, 1.1))
        with (patch("research.funding_spread_data._months", return_value=("2026-01",)),
              patch("research.funding_spread_data._pins", return_value=pins),
              patch("research.funding_spread_data._price_month", return_value=fake_prices)):
            rows = list(iter_hours())
        self.assertEqual(len(rows), 2)
        self.assertIsInstance(rows[0], HourBar)
        self.assertEqual(rows[0].time, first)
        self.assertIsInstance(rows[1], TerminalOpen)
        self.assertEqual(rows[1].time, first + timedelta(hours=1))
        self.assertEqual(rows[1].opening, (1.06,) * 3)

    def test_only_first_2026_funding_event_is_exposed(self):
        first = datetime(2026, 1, 1, tzinfo=UTC)
        pins = {("2026-01", symbol): "0" * 64 for symbol in SYMBOLS}
        def events(symbol, *_):
            return (FundingEvent(first, symbol, .001),
                    FundingEvent(first + timedelta(hours=8), symbol, .002))
        with (patch("research.funding_spread_data._months", return_value=("2026-01",)),
              patch("research.funding_spread_data._pins", return_value=pins),
              patch("research.funding_spread_data._funding_month", side_effect=events)):
            returned = load_funding()
        self.assertEqual(len(returned), 3)
        self.assertTrue(all(event.time == first for event in returned))

    def test_funding_interval_must_match_eight_hour_grid(self):
        start = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp() * 1000)
        lines = ["calc_time,funding_interval_hours,last_funding_rate"]
        for index in range(31 * 3):
            interval = 4 if index == 1 else 8
            lines.append(f"{start + index * 8 * 3_600_000},{interval},0.001")
        payload = zip_payload("ADAUSDT-fundingRate-2021-01.zip",
                              "\n".join(lines) + "\n")
        with patch("research.funding_spread_data._verified_zip",
                   side_effect=lambda *_: ZipFile(BytesIO(payload))):
            with self.assertRaisesRegex(FundingSpreadDataError, "funding grid or rate"):
                _funding_month("ADAUSDT", "2021-01", "0" * 64)


if __name__ == "__main__":
    unittest.main()
