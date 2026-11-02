"""Offline integrity tests for official USD-M archive ingestion."""

import json
import sys
import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.altcoin_reversal_archive import (
    ArchiveIntegrityError,
    _archive_url,
    _audit_bars,
    _audit_funding,
    _parse_bars,
    _parse_funding,
    load_and_audit,
)


BAR_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore\n"
)
FUNDING_HEADER = "calc_time,funding_interval_hours,last_funding_rate\n"


def _bar_row(moment, *, high=101.0, low=99.0, quote_volume=1000.0):
    open_ms = int(moment.timestamp() * 1000)
    close_ms = open_ms + 4 * 60 * 60 * 1000 - 1
    return (
        f"{open_ms},100,{high},{low},100,10,{close_ms},"
        f"{quote_volume},1,5,500,0\n"
    )


def _funding_row(moment, *, rate="0.0001"):
    return f"{int(moment.timestamp() * 1000)},8,{rate}\n"


def _zip_member(member_name, text):
    payload = BytesIO()
    with ZipFile(payload, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member_name, text)
    return payload.getvalue()


def _fixture_sources(*, mutate=None):
    symbol, month = "XTZUSDT", "2026-02"
    start = pd.Timestamp("2026-02-01", tz="UTC")
    end = pd.Timestamp("2026-03-01", tz="UTC")
    bar_times = pd.date_range(start, end - pd.Timedelta(hours=4), freq="4h")
    funding_times = pd.date_range(start, end - pd.Timedelta(hours=8), freq="8h")
    bars = BAR_HEADER + "".join(_bar_row(moment) for moment in bar_times)
    funding = FUNDING_HEADER + "".join(
        _funding_row(moment) for moment in funding_times
    )
    source = {}
    for kind, text in (
        ("klines", bars), ("markPriceKlines", bars),
        ("fundingRate", funding),
    ):
        url, basename = _archive_url(kind, symbol, month)
        member = basename[:-4] + ".csv"
        zipped = _zip_member(member, text)
        source[url] = zipped
        source[url + ".CHECKSUM"] = (
            f"{sha256(zipped).hexdigest()}  {basename}\n".encode("ascii")
        )
    if mutate:
        mutate(source)
    return source


class ArchiveIngestionTests(unittest.TestCase):
    def test_complete_month_is_verified_and_json_serializable(self):
        source = _fixture_sources()
        bundle = load_and_audit(
            symbols=("XTZUSDT",), start_month="2026-02", end_month="2026-02",
            max_workers=2, fetcher=source.__getitem__,
        )
        self.assertEqual(len(bundle.trade_bars["XTZUSDT"]), 168)
        self.assertEqual(len(bundle.mark_bars["XTZUSDT"]), 168)
        self.assertEqual(len(bundle.funding_events["XTZUSDT"]), 84)
        self.assertEqual(bundle.manifest["archive_count"], 3)
        self.assertEqual(bundle.manifest["coverage"]["XTZUSDT"]["trade"]["coverage"], 1.0)
        json.dumps(bundle.manifest, allow_nan=False)

    def test_checksum_mismatch_fails_closed(self):
        def mutate(source):
            url, _ = _archive_url("klines", "XTZUSDT", "2026-02")
            source[url] += b"garbage"

        source = _fixture_sources(mutate=mutate)
        with self.assertRaisesRegex(ArchiveIntegrityError, "SHA-256 mismatch"):
            load_and_audit(
                symbols=("XTZUSDT",), start_month="2026-02",
                end_month="2026-02", max_workers=1,
                fetcher=source.__getitem__,
            )

    def test_old_headerless_bar_keeps_first_row(self):
        moment = pd.Timestamp("2020-10-01", tz="UTC")
        frame = _parse_bars(
            _bar_row(moment).encode("ascii"), kind="klines",
            symbol="XTZUSDT", month="2020-10",
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.index[0], moment)

    def test_duplicate_bar_timestamps_are_rejected(self):
        moment = pd.Timestamp("2020-10-01", tz="UTC")
        with self.assertRaisesRegex(ArchiveIntegrityError, "duplicate"):
            _parse_bars(
                (_bar_row(moment) * 2).encode("ascii"), kind="klines",
                symbol="XTZUSDT", month="2020-10",
            )

    def test_invalid_ohlc_and_negative_quote_volume_are_rejected(self):
        moment = pd.Timestamp("2020-10-01", tz="UTC")
        with self.assertRaisesRegex(ArchiveIntegrityError, "inconsistent OHLC"):
            _parse_bars(
                _bar_row(moment, high=90).encode("ascii"), kind="klines",
                symbol="XTZUSDT", month="2020-10",
            )
        with self.assertRaisesRegex(ArchiveIntegrityError, "negative quote volume"):
            _parse_bars(
                _bar_row(moment, quote_volume=-1).encode("ascii"), kind="klines",
                symbol="XTZUSDT", month="2020-10",
            )

    def test_missing_bar_below_coverage_threshold_is_rejected(self):
        source = _fixture_sources()
        url, basename = _archive_url("klines", "XTZUSDT", "2026-02")
        text = BAR_HEADER + "".join(
            _bar_row(moment) for moment in pd.date_range(
                "2026-02-01", "2026-02-28 20:00", freq="4h", tz="UTC"
            ) if moment != pd.Timestamp("2026-02-02 04:00", tz="UTC")
        )
        zipped = _zip_member(basename[:-4] + ".csv", text)
        source[url] = zipped
        source[url + ".CHECKSUM"] = (
            f"{sha256(zipped).hexdigest()}  {basename}\n".encode("ascii")
        )
        with self.assertRaisesRegex(ArchiveIntegrityError, "coverage"):
            load_and_audit(
                symbols=("XTZUSDT",), start_month="2026-02",
                end_month="2026-02", max_workers=1,
                fetcher=source.__getitem__,
            )

    def test_rare_single_gap_allowed_but_two_consecutive_missing_rejected(self):
        start = pd.Timestamp("2021-01-01", tz="UTC")
        end = pd.Timestamp("2022-01-01", tz="UTC")
        full = pd.date_range(start, end - pd.Timedelta(hours=4), freq="4h",
                             name="open_time")
        base = pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "quote_volume": 1000.0,
        }, index=full)
        one_missing = base.drop(full[50])
        _, coverage = _audit_bars(
            [one_missing], kind="klines", symbol="XTZUSDT",
            start=start, end=end,
        )
        self.assertEqual(coverage["missing_bars"], 1)
        self.assertEqual(coverage["max_missing_run_4h_bars"], 1)
        with self.assertRaisesRegex(ArchiveIntegrityError, "gap exceeds 4 hours"):
            _audit_bars(
                [base.drop(full[[50, 51]])], kind="klines", symbol="XTZUSDT",
                start=start, end=end,
            )

    def test_partial_trade_and_two_prelisting_mark_bars_are_excluded(self):
        listing = pd.Timestamp("2020-10-16 08:00", tz="UTC")
        for kind, before in (("klines", 1), ("markPriceKlines", 2)):
            times = [listing - pd.Timedelta(hours=4 * step)
                     for step in range(before, 0, -1)] + [listing]
            frame = _parse_bars(
                "".join(_bar_row(moment) for moment in times).encode("ascii"),
                kind=kind, symbol="AAVEUSDT", month="2020-10",
            )
            checked, coverage = _audit_bars(
                [frame], kind=kind, symbol="AAVEUSDT",
                start=listing, end=listing + pd.Timedelta(hours=4),
            )
            self.assertEqual(checked.index[0], listing)
            self.assertEqual(coverage["prelisting_or_partial_bars_excluded"], before)

    def test_funding_preserves_millisecond_timestamp_and_rejects_long_gap(self):
        first = pd.Timestamp("2026-02-01 00:00:00.005", tz="UTC")
        second = first + pd.Timedelta(hours=32)
        events = _parse_funding(
            (FUNDING_HEADER + _funding_row(first) + _funding_row(second)).encode("ascii"),
            symbol="XTZUSDT", month="2026-02",
        )
        self.assertEqual(events["funding_time"].iloc[0], first)
        with self.assertRaisesRegex(ArchiveIntegrityError, "gap exceeds 24 hours"):
            _audit_funding(
                [events], symbol="XTZUSDT",
                start=pd.Timestamp("2026-02-01", tz="UTC"),
                end=pd.Timestamp("2026-02-03", tz="UTC"),
            )


if __name__ == "__main__":
    unittest.main()
