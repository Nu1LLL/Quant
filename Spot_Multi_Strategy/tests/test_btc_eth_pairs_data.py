"""Synthetic integrity tests for the pinned BTC/ETH feed; no live result read."""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from research.btc_eth_pairs_data import (
    FUNDING_MANIFEST_PATH,
    KLINE_MANIFEST_PATH,
    PairDataError,
    _funding_url,
    _read_funding_month,
    iter_paired_bars,
    read_funding_manifest,
    read_kline_manifest,
)


def _zipped(name: str, csv_text: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr(name, csv_text)
    return buffer.getvalue()


def _funding_fixture(*, bad_index: int | None = None,
                     bad_value: str | None = None,
                     interval_hours: str = "8") -> tuple[bytes, dict[str, str], object]:
    month, symbol = "2021-01", "ETHUSDT"
    url, basename = _funding_url(month, symbol)
    start_ms = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    lines = ["calc_time,funding_interval_hours,last_funding_rate"]
    for index in range(31 * 3):
        stamp = start_ms + index * 8 * 60 * 60 * 1000
        stamp += 2 if index == 0 else (1 if index == 92 else 0)
        rate = "0.0001"
        if index == bad_index:
            if bad_value == "off_anchor":
                stamp += 1000
            else:
                rate = bad_value or rate
        lines.append(f"{stamp},{interval_hours},{rate}")
    data = _zipped(basename[:-4] + ".csv", "\n".join(lines) + "\n")
    digest = sha256(data).hexdigest()
    pin = {"sha256": digest, "rows": "93"}
    mapping = {url: data, url + ".CHECKSUM": f"{digest}  {basename}\n".encode()}

    def fetch(source: str):
        payload = mapping[source]
        return (payload[index:index + 1024]
                for index in range(0, len(payload), 1024))

    return data, pin, fetch


class PinnedPairsFeedTests(unittest.TestCase):
    def test_manifests_require_fixed_two_symbol_history(self):
        self.assertEqual(len(read_kline_manifest()), 144)
        self.assertEqual(len(read_funding_manifest()), 120)
        self.assertIn(("2020-01", "ETHUSDT"), read_kline_manifest())
        self.assertIn(("2025-12", "BTCUSDT"), read_funding_manifest())
        raw = KLINE_MANIFEST_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "kline.csv"
            changed.write_text(raw.replace("2020-01,um,ETHUSDT", "2020-01,um,SOLUSDT", 1),
                               encoding="utf-8")
            with self.assertRaisesRegex(PairDataError, "invalid kline manifest pin"):
                read_kline_manifest(changed)
            changed.write_text(raw.replace("44640,0,0", "44639,0,0", 1),
                               encoding="utf-8")
            with self.assertRaisesRegex(PairDataError, "invalid kline manifest pin"):
                read_kline_manifest(changed)
        self.assertTrue(FUNDING_MANIFEST_PATH.is_file())

    def test_funding_preserves_actual_milliseconds_and_all_anchors(self):
        _, pin, fetch = _funding_fixture()
        events, digest = _read_funding_month("2021-01", "ETHUSDT", pin, fetch)
        self.assertEqual(len(events), 93)
        self.assertEqual(events[0].time.isoformat(), "2021-01-01T00:00:00.002000+00:00")
        self.assertEqual(events[-1].time.isoformat(), "2021-01-31T16:00:00.001000+00:00")
        self.assertEqual(events[0].symbol, "ETHUSDT")
        self.assertAlmostEqual(events[0].rate, 0.0001)
        self.assertEqual(digest, pin["sha256"])

    def test_funding_metadata_interval_does_not_override_actual_time_grid(self):
        _, pin, fetch = _funding_fixture(interval_hours="4")
        events, _ = _read_funding_month("2021-01", "ETHUSDT", pin, fetch)
        self.assertEqual(len(events), 93)
        self.assertEqual(events[0].time.hour, 0)
        self.assertEqual(events[1].time.hour, 8)

    def test_funding_rejects_off_anchor_nonfinite_and_pin_mismatch(self):
        _, pin, fetch = _funding_fixture(bad_index=8, bad_value="off_anchor")
        with self.assertRaisesRegex(PairDataError, "off anchor"):
            _read_funding_month("2021-01", "ETHUSDT", pin, fetch)
        _, pin, fetch = _funding_fixture(bad_index=8, bad_value="NaN")
        with self.assertRaisesRegex(PairDataError, "nonfinite"):
            _read_funding_month("2021-01", "ETHUSDT", pin, fetch)
        _, pin, fetch = _funding_fixture()
        pin["sha256"] = "0" * 64
        with self.assertRaisesRegex(PairDataError, "CHECKSUM or pin"):
            _read_funding_month("2021-01", "ETHUSDT", pin, fetch)

    def test_second_archive_pin_is_checked_before_first_price(self):
        pins = read_kline_manifest()

        @contextmanager
        def audit(request, **_kwargs):
            good_hash = pins[(request.month, request.symbol)]["sha256"]
            digest = good_hash if request.symbol == "BTCUSDT" else "0" * 64
            yield None, None, {"sha256": digest, "rows": 44640}

        with patch("research.btc_eth_pairs_data._opened_verified_archive", audit):
            iterator = iter_paired_bars(fetch_chunks=lambda _url: ())
            with self.assertRaisesRegex(PairDataError, "kline ZIP differs from pin"):
                next(iterator)

    def test_paired_bar_uses_same_utc_minute_and_exposes_audit_hashes(self):
        pins = read_kline_manifest()
        stamp = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

        class FakeArchive:
            def __init__(self, price: str):
                self.content = (
                    "open_time,open,high,low,close,volume,close_time,quote_volume,"
                    "count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
                    f"{stamp},{price},{price},{price},{price},1,"
                    f"{stamp + 59999},1,1,1,1,0\n"
                ).encode()

            def open(self, _member):
                return BytesIO(self.content)

        @contextmanager
        def audit(request, **_kwargs):
            yield (FakeArchive("100" if request.symbol == "BTCUSDT" else "10"),
                   "member", {
                       "sha256": pins[(request.month, request.symbol)]["sha256"],
                       "rows": 44640,
                   })

        hashes = {}
        with patch("research.btc_eth_pairs_data._opened_verified_archive", audit):
            iterator = iter_paired_bars(fetch_chunks=lambda _url: (), audit_hashes=hashes)
            first = next(iterator)
            iterator.close()
        self.assertEqual(first.time.isoformat(), "2020-01-01T00:00:00+00:00")
        self.assertEqual((first.btc_close, first.eth_close), (100.0, 10.0))
        self.assertEqual(hashes, {
            ("2020-01", symbol): pins[("2020-01", symbol)]["sha256"]
            for symbol in ("BTCUSDT", "ETHUSDT")
        })


if __name__ == "__main__":
    unittest.main()
