"""Synthetic-only audit tests; they never download test-period market outcomes."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import unittest
from zipfile import ZipFile

from research.network_adoption_binance import (
    ArchiveIntegrityError,
    ArchiveUnavailable,
    Bar,
    archive_url,
    bar_grid_gaps,
    fetch_and_parse_month,
    load_month,
)


UTC = timezone.utc


def _timestamp(dt, unit="ms"):
    microseconds = int((dt - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 1_000_000)
    return str(microseconds // 1000 if unit == "ms" else microseconds)


def _bar_row(opened, period, unit="ms", *, high="11", low="9", quote="100"):
    tick = timedelta(milliseconds=1) if unit == "ms" else timedelta(microseconds=1)
    closed = opened + period - tick
    return ",".join((
        _timestamp(opened, unit), "10", high, low, "10.5", "8",
        _timestamp(closed, unit), quote, "12", "4", "50", "0",
    ))


def _archive(kind="trade4", symbol="BATUSDT", month="2020-12", text=None,
             *, extra_member=False, checksum_override=None):
    url, basename = archive_url(kind, symbol, month)
    if text is None:
        opened = datetime(2020, 12, 1, tzinfo=UTC)
        text = _bar_row(opened, timedelta(hours=4)) + "\n"
    payload = BytesIO()
    with ZipFile(payload, "w") as zipped:
        zipped.writestr(basename[:-4] + ".csv", text)
        if extra_member:
            zipped.writestr("extra.txt", "x")
    zip_bytes = payload.getvalue()
    digest = checksum_override or sha256(zip_bytes).hexdigest()
    checksum = f"{digest}  {basename}\n".encode("ascii")
    def fetcher(requested_url):
        if requested_url == url + ".CHECKSUM":
            return checksum
        if requested_url == url:
            return zip_bytes
        raise OSError("unexpected URL")
    return fetcher, zip_bytes


class BinanceNetworkArchiveTests(unittest.TestCase):
    def test_urls_and_frozen_symbols(self):
        self.assertIn("/futures/um/monthly/klines/BATUSDT/4h/", archive_url(
            "trade4", "BATUSDT", "2020-12")[0])
        self.assertIn("/futures/um/monthly/markPriceKlines/BATUSDT/1h/", archive_url(
            "mark1", "BATUSDT", "2020-12")[0])
        self.assertIn("/futures/um/monthly/fundingRate/BATUSDT/", archive_url(
            "funding", "BATUSDT", "2020-12")[0])
        self.assertIn("/spot/monthly/klines/BTCUSDT/1d/", archive_url(
            "btc_spot1d", "BTCUSDT", "2020-12")[0])
        with self.assertRaises(ValueError):
            archive_url("trade4", "DOGEUSDT", "2020-12")
        with self.assertRaises(ValueError):
            archive_url("btc_spot1d", "ETHUSDT", "2020-12")

    def test_verified_trade_and_digest_manifest(self):
        fetcher, zip_bytes = _archive()
        result = fetch_and_parse_month("trade4", "BATUSDT", "2020-12", fetcher=fetcher)
        self.assertEqual(result.zip_bytes, len(zip_bytes))
        self.assertEqual(result.sha256, sha256(zip_bytes).hexdigest())
        self.assertEqual(result.rows[0].open_time, datetime(2020, 12, 1, tzinfo=UTC))
        self.assertEqual(result.rows[0].quote_volume, 100.0)
        self.assertEqual(result.rows, load_month("trade4", "BATUSDT", "2020-12", fetcher=fetcher))

    def test_mismatched_checksum_and_extra_zip_member_fail(self):
        fetcher, _ = _archive(checksum_override="0" * 64)
        with self.assertRaisesRegex(ArchiveIntegrityError, "SHA-256 mismatch"):
            load_month("trade4", "BATUSDT", "2020-12", fetcher=fetcher)
        fetcher, _ = _archive(extra_member=True)
        with self.assertRaisesRegex(ArchiveIntegrityError, "exactly"):
            load_month("trade4", "BATUSDT", "2020-12", fetcher=fetcher)

    def test_malformed_bar_rows_fail_closed(self):
        opened = datetime(2020, 12, 1, tzinfo=UTC)
        cases = (
            _bar_row(opened, timedelta(hours=4), high="9") + "\n",
            _bar_row(opened, timedelta(hours=4), quote="-1") + "\n",
            _bar_row(opened, timedelta(hours=4)) + ",extra\n",
            _bar_row(opened + timedelta(hours=1), timedelta(hours=4)) + "\n",
            _bar_row(opened, timedelta(hours=4)) + "\n" +
                _bar_row(opened, timedelta(hours=4)) + "\n",
            _bar_row(datetime(2021, 1, 1, tzinfo=UTC), timedelta(hours=4)) + "\n",
        )
        for case in cases:
            with self.subTest(case=case):
                fetcher, _ = _archive(text=case)
                with self.assertRaises(ArchiveIntegrityError):
                    load_month("trade4", "BATUSDT", "2020-12", fetcher=fetcher)

    def test_mark_hour_and_spot_microseconds(self):
        opened = datetime(2020, 12, 1, tzinfo=UTC)
        fetcher, _ = _archive(kind="mark1", text=_bar_row(
            opened, timedelta(hours=1)) + "\n")
        rows = load_month("mark1", "BATUSDT", "2020-12", fetcher=fetcher)
        self.assertIsNone(rows[0].quote_volume)
        fetcher, _ = _archive(kind="btc_spot1d", symbol="BTCUSDT", text=_bar_row(
            opened, timedelta(days=1), unit="us") + "\n")
        rows = load_month("btc_spot1d", "BTCUSDT", "2020-12", fetcher=fetcher)
        self.assertEqual(rows[0].quote_volume, 100.0)

    def test_funding_timestamp_and_duplicates(self):
        opened = datetime(2020, 12, 1, tzinfo=UTC)
        good = "calc_time,funding_interval_hours,last_funding_rate\n" + \
            f"{_timestamp(opened)},8,0.0001\n{_timestamp(opened+timedelta(hours=8))},8,-0.0002\n"
        fetcher, _ = _archive(kind="funding", text=good)
        rows = load_month("funding", "BATUSDT", "2020-12", fetcher=fetcher)
        self.assertEqual((rows[0].rate, rows[1].rate), (0.0001, -0.0002))
        jittered = f"{_timestamp(opened+timedelta(milliseconds=10))},8,0.0001\n"
        fetcher, _ = _archive(kind="funding", text=jittered)
        self.assertEqual(load_month("funding", "BATUSDT", "2020-12",
                                    fetcher=fetcher)[0].calc_time,
                         opened + timedelta(milliseconds=10))
        for bad in (
            good + f"{_timestamp(opened+timedelta(hours=8))},8,0.0001\n",
            f"{_timestamp(opened)},0,0.0001\n",
            f"{_timestamp(opened)},8,nan\n",
        ):
            with self.subTest(bad=bad):
                fetcher, _ = _archive(kind="funding", text=bad)
                with self.assertRaises(ArchiveIntegrityError):
                    load_month("funding", "BATUSDT", "2020-12", fetcher=fetcher)

    def test_archive_404_is_not_reinterpreted_as_delisting(self):
        def missing(_):
            raise OSError("404")
        with self.assertRaises(ArchiveUnavailable):
            load_month("trade4", "BATUSDT", "2020-12", fetcher=missing)

    def test_bar_grid_gaps_are_reported_not_excused(self):
        opened = datetime(2020, 12, 1, tzinfo=UTC)
        bars = (
            Bar(opened, 10, 11, 9, 10, 100),
            Bar(opened+timedelta(hours=8), 10, 11, 9, 10, 100),
        )
        gaps = bar_grid_gaps(bars, opened, opened+timedelta(hours=12), kind="trade4")
        self.assertEqual(gaps, (opened+timedelta(hours=4),))
        with self.assertRaises(ArchiveIntegrityError):
            bar_grid_gaps(bars + (bars[0],), opened, opened+timedelta(hours=12), kind="trade4")


if __name__ == "__main__":
    unittest.main()
