"""Offline-only integrity tests; no official price/funding result is read."""

import json
import sys
import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.altcoin_reversal_archive import _archive_url
from research.fed_reserves_crypto_archive import (
    ArchiveIntegrityError,
    SYMBOLS,
    _load_and_audit_impl,
)
from research.wiki_attention_daily_archive_data import _daily_archive_url


START = pd.Timestamp("2020-12-01", tz="UTC")
TERMINAL = pd.Timestamp("2021-01-01", tz="UTC")
BAR_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore\n"
)
FUNDING_HEADER = "calc_time,funding_interval_hours,last_funding_rate\n"


def _bar(moment, *, high="101", quote_volume="1000", microseconds=False):
    scale = 1_000_000 if microseconds else 1_000
    opened = int(moment.timestamp() * scale)
    closed = opened + 4 * 3600 * scale - 1
    return (
        f"{opened},100,{high},99,100,10,{closed},{quote_volume},"
        "1,5,500,0\n"
    )


def _funding(moment):
    return f"{int(moment.timestamp() * 1000)},8,0.0001\n"


def _put_zip(source, url, basename, text):
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(basename[:-4] + ".csv", text)
    source[url] = output.getvalue()
    source[url + ".CHECKSUM"] = (
        f"{sha256(output.getvalue()).hexdigest()}  {basename}\n".encode("ascii")
    )


def _fixture(*, symbols=("BTCUSDT",), missing_trade_day=None,
             missing_mark_day=None, missing_mark_bar=None,
             missing_funding=None, funding_jitter_ms=0,
             bad_terminal=False):
    source = {}
    bar_times = pd.date_range(
        START, TERMINAL - pd.Timedelta(hours=4), freq="4h", tz="UTC"
    )
    funding_times = pd.date_range(
        START, TERMINAL - pd.Timedelta(hours=8), freq="8h", tz="UTC"
    )
    for symbol in symbols:
        for kind, missing_day in (
            ("klines", missing_trade_day),
            ("markPriceKlines", missing_mark_day),
        ):
            rows = "".join(
                _bar(moment) for moment in bar_times
                if (missing_day is None or moment.normalize() != missing_day)
                and (kind != "markPriceKlines" or moment != missing_mark_bar)
            )
            url, basename = _archive_url(kind, symbol, "2020-12")
            _put_zip(source, url, basename, BAR_HEADER + rows)
            terminal_url, terminal_basename = _archive_url(
                kind, symbol, "2021-01"
            )
            terminal_start = (
                TERMINAL + pd.Timedelta(hours=4) if bad_terminal else TERMINAL
            )
            _put_zip(
                source, terminal_url, terminal_basename,
                BAR_HEADER + _bar(terminal_start, high="999999")
                + _bar(terminal_start + pd.Timedelta(hours=4), high="999999"),
            )
            if missing_day is not None:
                day_times = pd.date_range(missing_day, periods=6, freq="4h")
                day_url, day_basename = _daily_archive_url(kind, symbol, missing_day)
                _put_zip(
                    source, day_url, day_basename,
                    BAR_HEADER + "".join(_bar(moment) for moment in day_times),
                )
        funding_url, funding_basename = _archive_url(
            "fundingRate", symbol, "2020-12"
        )
        _put_zip(
            source, funding_url, funding_basename,
            FUNDING_HEADER + "".join(
                _funding(moment + pd.Timedelta(milliseconds=funding_jitter_ms))
                for moment in funding_times if moment != missing_funding
            ),
        )
        terminal_url, terminal_basename = _archive_url(
            "fundingRate", symbol, "2021-01"
        )
        _put_zip(
            source, terminal_url, terminal_basename,
            FUNDING_HEADER + _funding(
                TERMINAL + pd.Timedelta(milliseconds=funding_jitter_ms)
            ) + _funding(TERMINAL + pd.Timedelta(hours=8)),
        )
    return source


def _load(source, *, symbols=("BTCUSDT",), fetcher=None):
    return _load_and_audit_impl(
        symbols=symbols, start=START, terminal=TERMINAL,
        last_archive_month="2020-12",
        fetcher=fetcher or source.__getitem__, max_workers=2,
    )


class FedReservesCryptoArchiveTests(unittest.TestCase):
    def test_preregistered_symbol_universe_is_fixed(self):
        self.assertEqual(
            SYMBOLS,
            ("BTCUSDT", "ETHUSDT", "DOGEUSDT", "LTCUSDT", "BCHUSDT"),
        )

    def test_complete_fixture_has_no_daily_or_wiki_requests(self):
        source = _fixture(symbols=SYMBOLS)
        requested = []

        def fetch(url):
            requested.append(url)
            return source[url]

        bundle = _load(source, symbols=SYMBOLS, fetcher=fetch)
        self.assertEqual(set(bundle.trade_bars), set(SYMBOLS))
        self.assertEqual(len(bundle.trade_bars["BTCUSDT"]), 187)
        self.assertEqual(len(bundle.funding_events["BTCUSDT"]), 94)
        self.assertEqual(bundle.manifest["monthly_archive_count"], 30)
        self.assertEqual(bundle.manifest["daily_archive_count"], 0)
        self.assertFalse(any("wikimedia" in url for url in requested))
        self.assertFalse(any("/daily/" in url for url in requested))
        self.assertTrue(pd.isna(bundle.mark_bars["BTCUSDT"].loc[TERMINAL, "high"]))
        self.assertTrue(pd.isna(bundle.trade_bars["BTCUSDT"].loc[TERMINAL, "close"]))
        self.assertEqual(bundle.trade_bars["BTCUSDT"].loc[TERMINAL, "open"], 100)
        json.dumps(bundle.manifest, allow_nan=False)

    def test_exact_whole_day_repair_and_fingerprint(self):
        trade_day = pd.Timestamp("2020-12-02", tz="UTC")
        mark_day = pd.Timestamp("2020-12-03", tz="UTC")
        bundle = _load(_fixture(
            missing_trade_day=trade_day, missing_mark_day=mark_day
        ))
        self.assertEqual(len(bundle.trade_bars["BTCUSDT"]), 187)
        self.assertEqual(len(bundle.mark_bars["BTCUSDT"]), 187)
        self.assertEqual(bundle.manifest["monthly_archive_count"], 6)
        self.assertEqual(bundle.manifest["daily_archive_count"], 2)
        self.assertEqual(bundle.manifest["archive_count"], 8)
        self.assertEqual(bundle.manifest["daily_replacements"], [
            {"kind": "klines", "symbol": "BTCUSDT",
             "month": "2020-12", "date": "2020-12-02"},
            {"kind": "markPriceKlines", "symbol": "BTCUSDT",
             "month": "2020-12", "date": "2020-12-03"},
        ])
        archive = bundle.manifest["archives"]["daily/klines/BTCUSDT/2020-12-02"]
        self.assertEqual(archive["rows_read"], 6)
        self.assertEqual(len(archive["sha256"]), 64)
        self.assertEqual(len(archive["checksum_sha256"]), 64)

    def test_partial_day_rejected_before_any_daily_fetch(self):
        source = _fixture(
            missing_mark_bar=pd.Timestamp("2020-12-02 04:00", tz="UTC")
        )
        requested = []

        def fetch(url):
            requested.append(url)
            return source[url]

        with self.assertRaisesRegex(ArchiveIntegrityError, "partial UTC day"):
            _load(source, fetcher=fetch)
        self.assertFalse(any("/daily/" in url for url in requested))

    def test_checksum_mismatch_rejected(self):
        source = _fixture()
        url, _ = _archive_url("markPriceKlines", "BTCUSDT", "2020-12")
        source[url] += b"corrupted"
        with self.assertRaisesRegex(ArchiveIntegrityError, "SHA-256 mismatch"):
            _load(source)

    def test_funding_gap_and_clock_offset_rejected(self):
        source = _fixture(
            missing_funding=pd.Timestamp("2020-12-02 08:00", tz="UTC")
        )
        with self.assertRaisesRegex(ArchiveIntegrityError, "missing or mistimed funding"):
            _load(source)
        with self.assertRaisesRegex(ArchiveIntegrityError, "off 4h grid"):
            _load(_fixture(funding_jitter_ms=101))

    def test_bounded_funding_jitter_preserves_raw_timestamp(self):
        bundle = _load(_fixture(funding_jitter_ms=33))
        event = bundle.funding_events["BTCUSDT"].iloc[0]
        self.assertEqual(event["funding_time"], START)
        self.assertEqual(event["raw_funding_time"], START + pd.Timedelta(milliseconds=33))

    def test_missing_terminal_open_rejected(self):
        with self.assertRaisesRegex(ArchiveIntegrityError, "terminal 00:00 bar"):
            _load(_fixture(bad_terminal=True))

    def test_invalid_daily_ohlc_and_checksum_rejected(self):
        day = pd.Timestamp("2020-12-02", tz="UTC")
        source = _fixture(missing_mark_day=day)
        url, basename = _daily_archive_url("markPriceKlines", "BTCUSDT", day)
        source[url] += b"corrupted"
        with self.assertRaisesRegex(ArchiveIntegrityError, "SHA-256 mismatch"):
            _load(source)
        source = _fixture(missing_mark_day=day)
        rows = _bar(day, high="98") + "".join(
            _bar(moment) for moment in pd.date_range(
                day + pd.Timedelta(hours=4), periods=5, freq="4h"
            )
        )
        _put_zip(source, url, basename, BAR_HEADER + rows)
        with self.assertRaisesRegex(ArchiveIntegrityError, "high below"):
            _load(source)


if __name__ == "__main__":
    unittest.main()
