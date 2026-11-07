"""Synthetic input-integrity tests for the frozen three-altcoin flow feed."""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from research.alt_common_flow_data import (
    AltFlowDataError, FUNDING_MANIFEST, KLINE_MANIFEST, SYMBOLS,
    _funding_month, iter_minutes, read_manifest,
)


class AltFlowFeedTests(unittest.TestCase):
    def test_fixed_manifests_cover_three_symbols_and_five_oos_years(self):
        k = read_manifest(KLINE_MANIFEST, kind="kline")
        f = read_manifest(FUNDING_MANIFEST, kind="funding")
        self.assertEqual(len(k), 192)
        self.assertEqual(len(f), 180)
        self.assertEqual({s for _, s in k}, set(SYMBOLS))
        self.assertEqual({s for _, s in f}, set(SYMBOLS))

    def test_third_symbol_hash_is_checked_before_any_price_is_yielded(self):
        pins = read_manifest(KLINE_MANIFEST, kind="kline")

        @contextmanager
        def fake_archive(request, **_kwargs):
            digest = pins[(request.month, request.symbol)]
            if request.symbol == "DOGEUSDT":
                digest = "0" * 64
            yield None, None, {"sha256": digest}

        with patch("research.alt_common_flow_data._opened_verified_archive", fake_archive):
            stream = iter_minutes(fetch_chunks=lambda _url: ())
            with self.assertRaisesRegex(AltFlowDataError, "hash differs"):
                next(stream)
            stream.close()

    def test_funding_uses_actual_ms_and_rejects_off_anchor(self):
        month, symbol = "2021-01", "DOGEUSDT"
        name = f"{symbol}-fundingRate-{month}.zip"
        url = ("https://data.binance.vision/data/futures/um/monthly/fundingRate/"
               f"{symbol}/{name}")
        first = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

        def fixture(*, bad_index=None):
            lines = ["calc_time,funding_interval_hours,last_funding_rate"]
            for i in range(93):
                stamp = first + i * 8 * 3600 * 1000 + (2 if i == 0 else 0)
                if i == bad_index:
                    stamp += 1000
                lines.append(f"{stamp},8,0.0001")
            buffer = BytesIO()
            with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
                archive.writestr(name[:-4] + ".csv", "\n".join(lines) + "\n")
            payload = buffer.getvalue()
            digest = sha256(payload).hexdigest()
            mapping = {url: payload, url + ".CHECKSUM":
                       f"{digest}  {name}\n".encode()}
            return digest, lambda address: (mapping[address],)

        digest, fetch = fixture()
        events, actual = _funding_month(month, symbol, digest, fetch)
        self.assertEqual(len(events), 93)
        self.assertEqual(actual, digest)
        self.assertEqual(events[0].time.microsecond, 2000)
        digest, fetch = fixture(bad_index=4)
        with self.assertRaisesRegex(AltFlowDataError, "anchor"):
            _funding_month(month, symbol, digest, fetch)


if __name__ == "__main__":
    unittest.main()
