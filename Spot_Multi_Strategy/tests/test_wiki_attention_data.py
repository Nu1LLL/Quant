"""Deterministic offline integrity tests for the preregistered source loader."""

import json
import sys
import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.wiki_attention_data import (
    ArchiveIntegrityError,
    _archive_url,
    _load_and_audit_impl,
    _page_url,
    _pageviews,
    _parse_complete_bars,
)


BAR_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore\n"
)
FUNDING_HEADER = "calc_time,funding_interval_hours,last_funding_rate\n"


def _bar_row(moment, *, microseconds=False, high="101", low="99"):
    scale = 1_000_000 if microseconds else 1_000
    opened = int(moment.timestamp() * scale)
    closed = opened + 4 * 3600 * scale - 1
    return f"{opened},100,{high},{low},100,10,{closed},1000,1,5,500,0\n"


def _funding_row(moment, *, rate="0.0001"):
    return f"{int(moment.timestamp() * 1000)},8,{rate}\n"


def _zip(member, text):
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return output.getvalue()


def _set_archive(source, kind, month, text):
    url, basename = _archive_url(kind, "BTCUSDT", month)
    zipped = _zip(basename[:-4] + ".csv", text)
    source[url] = zipped
    source[url + ".CHECKSUM"] = (
        f"{sha256(zipped).hexdigest()}  {basename}\n".encode("ascii")
    )


def _fixture(*, missing_bar=None, missing_funding=None, extra_page=None,
             funding_jitter_ms=0):
    start = pd.Timestamp("2020-12-01", tz="UTC")
    terminal = pd.Timestamp("2021-01-01", tz="UTC")
    days = pd.date_range(start, terminal - pd.Timedelta(days=1), freq="D")
    items = [{
        "project": "en.wikipedia", "article": "Bitcoin",
        "granularity": "daily", "access": "all-access", "agent": "user",
        "timestamp": f"{day:%Y%m%d}00", "views": 100 + index,
    } for index, day in enumerate(days)]
    if extra_page is not None:
        extra_page(items)
    source = {
        _page_url("Bitcoin", start, terminal - pd.Timedelta(days=1)):
        json.dumps({"items": items}).encode("utf-8")
    }
    bar_times = pd.date_range(start, terminal - pd.Timedelta(hours=4), freq="4h")
    funding_times = pd.date_range(start, terminal - pd.Timedelta(hours=8), freq="8h")
    december_bars = BAR_HEADER + "".join(
        _bar_row(t) for t in bar_times if t != missing_bar
    )
    december_funding = FUNDING_HEADER + "".join(
        _funding_row(t + pd.Timedelta(milliseconds=funding_jitter_ms))
        for t in funding_times if t != missing_funding
    )
    for kind, text in (
        ("klines", december_bars),
        ("markPriceKlines", december_bars),
        ("fundingRate", december_funding),
    ):
        _set_archive(source, kind, "2020-12", text)
    # The 2021-01 archive deliberately contains later data. A valid loader
    # must not parse any post-terminal OHLC or funding value.
    january_bars = (
        BAR_HEADER + _bar_row(terminal, high="999999", low="0.0001")
        + _bar_row(terminal + pd.Timedelta(hours=4), high="999999")
    )
    january_funding = (
        FUNDING_HEADER + _funding_row(terminal + pd.Timedelta(milliseconds=funding_jitter_ms))
        + _funding_row(terminal + pd.Timedelta(hours=8), rate="999")
    )
    for kind, text in (
        ("klines", january_bars),
        ("markPriceKlines", january_bars),
        ("fundingRate", january_funding),
    ):
        _set_archive(source, kind, "2021-01", text)
    return source


def _load(source):
    return _load_and_audit_impl(
        titles={"BTCUSDT": "Bitcoin"},
        start=pd.Timestamp("2020-12-01", tz="UTC"),
        page_end=pd.Timestamp("2020-12-31", tz="UTC"),
        terminal=pd.Timestamp("2021-01-01", tz="UTC"),
        last_archive_month="2020-12", fetcher=source.__getitem__,
        max_workers=2,
    )


class WikipediaAttentionDataTests(unittest.TestCase):
    def test_complete_sources_are_verified_and_terminal_is_open_only(self):
        bundle = _load(_fixture())
        self.assertEqual(len(bundle.views["BTCUSDT"]), 31)
        self.assertEqual(len(bundle.trade_bars["BTCUSDT"]), 187)
        self.assertEqual(len(bundle.funding_events["BTCUSDT"]), 94)
        terminal = pd.Timestamp("2021-01-01", tz="UTC")
        self.assertEqual(bundle.trade_bars["BTCUSDT"].loc[terminal, "open"], 100)
        self.assertTrue(pd.isna(bundle.trade_bars["BTCUSDT"].loc[terminal, "high"]))
        self.assertTrue(pd.isna(bundle.mark_bars["BTCUSDT"].loc[terminal, "low"]))
        self.assertEqual(bundle.funding_events["BTCUSDT"]["funding_time"].iloc[-1], terminal)
        self.assertEqual(bundle.manifest["archive_count"], 6)
        self.assertEqual(bundle.manifest["wikipedia"]["BTCUSDT"]["http_status"], 200)
        self.assertTrue(bundle.manifest["archives"]["klines/BTCUSDT/2021-01"]["terminal_only"])
        json.dumps(bundle.manifest, allow_nan=False)

    def test_pageviews_reject_missing_duplicate_zero_and_dimensions(self):
        start = pd.Timestamp("2020-12-01", tz="UTC")
        end = pd.Timestamp("2020-12-31", tz="UTC")
        for mutate, error in (
            (lambda rows: rows.pop(3), "incomplete daily grid"),
            (lambda rows: rows.append(rows[3].copy()), "duplicate daily"),
            (lambda rows: rows[3].update(views=0), "positive integers"),
            (lambda rows: rows[3].update(article="Other"), "unexpected pageview dimensions"),
        ):
            with self.subTest(error=error):
                source = _fixture(extra_page=mutate)
                body = source[_page_url("Bitcoin", start, end)]
                with self.assertRaisesRegex(ArchiveIntegrityError, error):
                    _pageviews(body, title="Bitcoin", start=start, end=end)

    def test_archive_checksum_mismatch_fails_closed(self):
        source = _fixture()
        url, _ = _archive_url("klines", "BTCUSDT", "2020-12")
        source[url] += b"trailing garbage"
        with self.assertRaisesRegex(ArchiveIntegrityError, "SHA-256 mismatch"):
            _load(source)

    def test_missing_single_bar_fails_strict_grid(self):
        source = _fixture(missing_bar=pd.Timestamp("2020-12-02 04:00", tz="UTC"))
        with self.assertRaisesRegex(ArchiveIntegrityError, "incomplete 4h grid"):
            _load(source)

    def test_missing_single_funding_event_fails_strict_timing(self):
        source = _fixture(missing_funding=pd.Timestamp("2020-12-02 08:00", tz="UTC"))
        with self.assertRaisesRegex(ArchiveIntegrityError, "missing or mistimed funding"):
            _load(source)

    def test_funding_clock_jitter_is_bounded_and_raw_time_preserved(self):
        bundle = _load(_fixture(funding_jitter_ms=33))
        events = bundle.funding_events["BTCUSDT"]
        start = pd.Timestamp("2020-12-01", tz="UTC")
        self.assertEqual(events["funding_time"].iloc[0], start)
        self.assertEqual(events["raw_funding_time"].iloc[0], start + pd.Timedelta(milliseconds=33))
        self.assertEqual(bundle.manifest["archives"]["fundingRate/BTCUSDT/2020-12"]["max_clock_offset_ms"], 33)
        with self.assertRaisesRegex(ArchiveIntegrityError, "off 4h grid"):
            _load(_fixture(funding_jitter_ms=101))

    def test_microsecond_close_time_is_unit_aware_and_still_strict(self):
        moment = pd.Timestamp("2025-01-01", tz="UTC")
        valid = _parse_complete_bars(
            (BAR_HEADER + _bar_row(moment, microseconds=True)).encode("ascii"),
            kind="klines", symbol="BTCUSDT", month="2025-01",
        )
        self.assertEqual(valid.index[0], moment)
        wrong = _bar_row(moment, microseconds=True).split(",")
        wrong[6] = str(int(wrong[6]) - 999)  # 1 millisecond rather than 1 microsecond.
        with self.assertRaisesRegex(ArchiveIntegrityError, "close times"):
            _parse_complete_bars(
                (BAR_HEADER + ",".join(wrong)).encode("ascii"), kind="klines",
                symbol="BTCUSDT", month="2025-01",
            )


if __name__ == "__main__":
    unittest.main()
