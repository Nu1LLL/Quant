"""Offline-only tests for the separately registered daily-archive repair."""

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
from research.wiki_attention_data import ArchiveIntegrityError, _page_url
from research.wiki_attention_daily_archive_data import (
    _daily_archive_url,
    _load_and_audit_impl,
    _load_daily_archive,
)


START = pd.Timestamp("2020-12-01", tz="UTC")
END = pd.Timestamp("2021-01-01", tz="UTC")
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


def _archive(source, url, basename, text):
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(basename[:-4] + ".csv", text)
    source[url] = buffer.getvalue()
    source[url + ".CHECKSUM"] = (
        f"{sha256(buffer.getvalue()).hexdigest()}  {basename}\n".encode("ascii")
    )


def _fixture(*, missing_trade_day=None, missing_mark_day=None,
             missing_mark_bar=None, daily_mark_rows=None):
    days = pd.date_range(START, END - pd.Timedelta(days=1), freq="D")
    items = [
        {"project": "en.wikipedia", "article": "Bitcoin",
         "granularity": "daily", "access": "all-access", "agent": "user",
         "timestamp": f"{day:%Y%m%d}00", "views": 100 + n}
        for n, day in enumerate(days)
    ]
    source = {
        _page_url("Bitcoin", START, END - pd.Timedelta(days=1)):
        json.dumps({"items": items}).encode("utf-8")
    }
    bars = pd.date_range(START, END - pd.Timedelta(hours=4), freq="4h")
    for kind, missing_day in (
        ("klines", missing_trade_day),
        ("markPriceKlines", missing_mark_day),
    ):
        monthly_rows = [
            _bar(moment) for moment in bars
            if (missing_day is None or moment.normalize() != missing_day)
            and (kind != "markPriceKlines" or moment != missing_mark_bar)
        ]
        url, basename = _archive_url(kind, "BTCUSDT", "2020-12")
        _archive(source, url, basename, BAR_HEADER + "".join(monthly_rows))
        jan_url, jan_basename = _archive_url(kind, "BTCUSDT", "2021-01")
        # Later rows have values that must remain unread. Only the exact
        # terminal 00:00 open may enter the returned bundle.
        _archive(
            source, jan_url, jan_basename,
            BAR_HEADER + _bar(END, high="999999")
            + _bar(END + pd.Timedelta(hours=4), high="999999"),
        )
        if missing_day is not None:
            daily_times = pd.date_range(missing_day, periods=6, freq="4h")
            rows = "".join(_bar(moment) for moment in daily_times)
            if kind == "markPriceKlines" and daily_mark_rows is not None:
                rows = daily_mark_rows(daily_times)
            daily_url, daily_basename = _daily_archive_url(
                kind, "BTCUSDT", missing_day
            )
            _archive(source, daily_url, daily_basename, BAR_HEADER + rows)
    monthly_funding = FUNDING_HEADER + "".join(
        _funding(moment) for moment in pd.date_range(
            START, END - pd.Timedelta(hours=8), freq="8h"
        )
    )
    funding_url, funding_basename = _archive_url(
        "fundingRate", "BTCUSDT", "2020-12"
    )
    _archive(source, funding_url, funding_basename, monthly_funding)
    jan_url, jan_basename = _archive_url(
        "fundingRate", "BTCUSDT", "2021-01"
    )
    _archive(
        source, jan_url, jan_basename,
        FUNDING_HEADER + _funding(END) + _funding(END + pd.Timedelta(hours=8)),
    )
    return source


def _load(source):
    return _load_and_audit_impl(
        titles={"BTCUSDT": "Bitcoin"}, start=START,
        page_end=END - pd.Timedelta(days=1), terminal=END,
        last_archive_month="2020-12", fetcher=source.__getitem__,
        max_workers=2,
    )


class DailyArchiveDataTests(unittest.TestCase):
    def test_complete_month_never_requests_daily_sources(self):
        source = _fixture()
        requested = []

        def fetch(url):
            requested.append(url)
            return source[url]

        bundle = _load_and_audit_impl(
            titles={"BTCUSDT": "Bitcoin"}, start=START,
            page_end=END - pd.Timedelta(days=1), terminal=END,
            last_archive_month="2020-12", fetcher=fetch, max_workers=2,
        )
        self.assertEqual(len(bundle.trade_bars["BTCUSDT"]), 187)
        self.assertEqual(bundle.manifest["daily_archive_count"], 0)
        self.assertEqual(bundle.manifest["daily_replacements"], [])
        self.assertFalse(any("/futures/um/daily/" in url for url in requested))

    def test_whole_day_repair_is_exact_per_kind_and_fingerprinted(self):
        trade_day = pd.Timestamp("2020-12-02", tz="UTC")
        mark_day = pd.Timestamp("2020-12-03", tz="UTC")
        bundle = _load(_fixture(
            missing_trade_day=trade_day, missing_mark_day=mark_day
        ))
        self.assertEqual(len(bundle.trade_bars["BTCUSDT"]), 187)
        self.assertEqual(len(bundle.mark_bars["BTCUSDT"]), 187)
        self.assertEqual(len(bundle.funding_events["BTCUSDT"]), 94)
        self.assertEqual(bundle.manifest["archive_count"], 8)
        self.assertEqual(bundle.manifest["monthly_archive_count"], 6)
        self.assertEqual(bundle.manifest["daily_archive_count"], 2)
        self.assertEqual(
            bundle.manifest["daily_replacements"],
            [
                {"kind": "klines", "symbol": "BTCUSDT",
                 "month": "2020-12", "date": "2020-12-02"},
                {"kind": "markPriceKlines", "symbol": "BTCUSDT",
                 "month": "2020-12", "date": "2020-12-03"},
            ],
        )
        daily = bundle.manifest["archives"]["daily/klines/BTCUSDT/2020-12-02"]
        self.assertEqual(daily["rows_read"], 6)
        self.assertEqual(daily["replacement_date"], "2020-12-02")
        self.assertEqual(len(daily["sha256"]), 64)
        self.assertTrue(daily["url"].endswith("BTCUSDT-4h-2020-12-02.zip"))
        self.assertTrue(pd.isna(bundle.trade_bars["BTCUSDT"].loc[END, "high"]))
        json.dumps(bundle.manifest, allow_nan=False)

    def test_partial_day_fails_before_daily_fetch(self):
        source = _fixture(
            missing_mark_bar=pd.Timestamp("2020-12-02 04:00", tz="UTC")
        )
        requested = []

        def fetch(url):
            requested.append(url)
            return source[url]

        with self.assertRaisesRegex(ArchiveIntegrityError, "partial UTC day"):
            _load_and_audit_impl(
                titles={"BTCUSDT": "Bitcoin"}, start=START,
                page_end=END - pd.Timedelta(days=1), terminal=END,
                last_archive_month="2020-12", fetcher=fetch,
                max_workers=2,
            )
        self.assertFalse(any("/futures/um/daily/" in url for url in requested))

    def test_daily_checksum_mismatch_fails_closed(self):
        day = pd.Timestamp("2020-12-02", tz="UTC")
        source = _fixture(missing_mark_day=day)
        url, _ = _daily_archive_url("markPriceKlines", "BTCUSDT", day)
        source[url] += b"corruption"
        with self.assertRaisesRegex(ArchiveIntegrityError, "SHA-256 mismatch"):
            _load(source)

    def test_missing_or_extra_daily_bar_fails_closed(self):
        day = pd.Timestamp("2020-12-02", tz="UTC")
        for mutate in (
            lambda times: "".join(_bar(moment) for moment in times[:-1]),
            lambda times: "".join(_bar(moment) for moment in times)
            + _bar(day + pd.Timedelta(days=1)),
        ):
            with self.subTest(mutate=mutate):
                source = _fixture(
                    missing_mark_day=day, daily_mark_rows=mutate
                )
                with self.assertRaises(ArchiveIntegrityError):
                    _load(source)

    def test_daily_ohlc_and_quote_volume_validation(self):
        day = pd.Timestamp("2020-12-02", tz="UTC")
        for kind, row in (
            ("markPriceKlines", _bar(day, high="98")),
            ("klines", _bar(day, quote_volume="-1")),
        ):
            with self.subTest(kind=kind):
                source = _fixture()
                url, basename = _daily_archive_url(kind, "BTCUSDT", day)
                rows = row + "".join(
                    _bar(moment) for moment in pd.date_range(
                        day + pd.Timedelta(hours=4), periods=5, freq="4h"
                    )
                )
                _archive(source, url, basename, BAR_HEADER + rows)
                with self.assertRaises(ArchiveIntegrityError):
                    _load_daily_archive(kind, "BTCUSDT", day, source.__getitem__)

    def test_microsecond_daily_close_time_is_accepted_if_exact(self):
        day = pd.Timestamp("2020-12-02", tz="UTC")
        source = _fixture()
        url, basename = _daily_archive_url("klines", "BTCUSDT", day)
        rows = "".join(
            _bar(moment, microseconds=True)
            for moment in pd.date_range(day, periods=6, freq="4h")
        )
        _archive(source, url, basename, BAR_HEADER + rows)
        frame, detail = _load_daily_archive(
            "klines", "BTCUSDT", day, source.__getitem__
        )
        self.assertEqual(len(frame), 6)
        self.assertEqual(detail["rows_read"], 6)

    def test_daily_url_rejects_funding_and_non_midnight(self):
        with self.assertRaisesRegex(ValueError, "prohibited"):
            _daily_archive_url("fundingRate", "BTCUSDT", START)
        with self.assertRaisesRegex(ValueError, "UTC midnight"):
            _daily_archive_url(
                "klines", "BTCUSDT", START + pd.Timedelta(hours=1)
            )


if __name__ == "__main__":
    unittest.main()
