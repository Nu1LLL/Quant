"""Synthetic-only integrity tests for the frozen spot-flow data loader."""

import json
import sys
import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.spot_flow_archive import (
    ArchiveIntegrityError,
    _archive_url,
    _parse_bars,
    _parse_funding,
    load_and_audit,
)


BAR_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore\n"
)
FUNDING_HEADER = "calc_time,funding_interval_hours,last_funding_rate\n"


def _bar_row(moment, *, spot=False, quote=1000.0, taker=600.0,
             high=101.0, volume=10.0):
    unit = 1_000_000 if spot else 1000
    opened = int(moment.timestamp() * unit)
    closed = opened + 4 * 60 * 60 * unit - 1
    return (
        f"{opened},100,{high},99,100,{volume},{closed},{quote},1,"
        f"5,{taker},0\n"
    )


def _funding_row(moment, rate="0.0001"):
    return f"{int(moment.timestamp() * 1000)},8,{rate}\n"


def _zip_csv(member_name, text):
    payload = BytesIO()
    with ZipFile(payload, "w", compression=ZIP_DEFLATED) as zipped:
        zipped.writestr(member_name, text)
    return payload.getvalue()


def _install(source, kind, symbol, month, text):
    url, basename = _archive_url(kind, symbol, month)
    zipped = _zip_csv(basename[:-4] + ".csv", text)
    source[url] = zipped
    source[url + ".CHECKSUM"] = (
        f"{sha256(zipped).hexdigest()}  {basename}\n".encode("ascii")
    )


def _fixture_sources(*, alter=None):
    symbol, month = "BTCUSDT", "2026-02"
    start = pd.Timestamp("2026-02-01", tz="UTC")
    end = pd.Timestamp("2026-03-01", tz="UTC")
    bars = pd.date_range(start, end - pd.Timedelta(hours=4), freq="4h")
    funding = pd.date_range(start, end - pd.Timedelta(hours=8), freq="8h")
    rows = {
        "spot": BAR_HEADER + "".join(_bar_row(t, spot=True) for t in bars),
        "trade": BAR_HEADER + "".join(_bar_row(t) for t in bars),
        "mark": BAR_HEADER + "".join(_bar_row(t) for t in bars),
        "funding": FUNDING_HEADER + "".join(_funding_row(t) for t in funding),
    }
    if alter:
        alter(rows, bars, funding)
    source = {}
    for kind, text in rows.items():
        _install(source, kind, symbol, month, text)
    return source


def _load(source):
    return load_and_audit(
        symbols=("BTCUSDT",), start_month="2026-02",
        end_month="2026-02", max_workers=2, fetcher=source.__getitem__,
    )


class SpotFlowArchiveTests(unittest.TestCase):
    def test_month_has_verified_spot_trade_mark_and_funding(self):
        bundle = _load(_fixture_sources())
        self.assertEqual(len(bundle.spot_bars["BTCUSDT"]), 168)
        self.assertEqual(len(bundle.trade_bars["BTCUSDT"]), 168)
        self.assertEqual(len(bundle.mark_bars["BTCUSDT"]), 168)
        self.assertEqual(len(bundle.funding_events["BTCUSDT"]), 84)
        self.assertEqual(
            list(bundle.spot_bars["BTCUSDT"].columns),
            ["open", "high", "low", "close", "quote_volume",
             "taker_buy_quote_volume"],
        )
        self.assertEqual(list(bundle.mark_bars["BTCUSDT"].columns),
                         ["open", "high", "low", "close"])
        self.assertEqual(bundle.manifest["archive_count"], 4)
        for kind in ("spot", "trade", "mark"):
            self.assertEqual(bundle.manifest["coverage"]["BTCUSDT"][kind]["coverage"], 1.0)
        json.dumps(bundle.manifest, allow_nan=False)

    def test_ms_and_us_close_time_boundaries(self):
        old = pd.Timestamp("2024-12-31 20:00", tz="UTC")
        new = pd.Timestamp("2025-01-01 00:00", tz="UTC")
        old_spot = _parse_bars(_bar_row(old).encode(), kind="spot",
                               symbol="BTCUSDT", month="2024-12")
        new_spot = _parse_bars(_bar_row(new, spot=True).encode(), kind="spot",
                               symbol="BTCUSDT", month="2025-01")
        trade = _parse_bars(_bar_row(new).encode(), kind="trade",
                            symbol="BTCUSDT", month="2025-01")
        self.assertEqual(old_spot.index[0], old)
        self.assertEqual(new_spot.index[0], new)
        self.assertEqual(trade.index[0], new)

    def test_wrong_timestamp_units_are_rejected_by_dataset_and_month(self):
        old = pd.Timestamp("2024-12-01", tz="UTC")
        new = pd.Timestamp("2025-01-01", tz="UTC")
        with self.assertRaisesRegex(ArchiveIntegrityError, "expected milliseconds"):
            _parse_bars(_bar_row(old, spot=True).encode(), kind="spot",
                        symbol="BTCUSDT", month="2024-12")
        with self.assertRaisesRegex(ArchiveIntegrityError, "expected microseconds"):
            _parse_bars(_bar_row(new).encode(), kind="spot",
                        symbol="BTCUSDT", month="2025-01")
        for kind in ("trade", "mark"):
            with self.subTest(kind=kind), self.assertRaisesRegex(
                    ArchiveIntegrityError, "expected milliseconds"):
                _parse_bars(_bar_row(new, spot=True).encode(), kind=kind,
                            symbol="BTCUSDT", month="2025-01")
        funding_us = f"{int(new.timestamp() * 1_000_000)},8,0.0001\n"
        with self.assertRaisesRegex(ArchiveIntegrityError, "expected milliseconds"):
            _parse_funding(funding_us.encode(), symbol="BTCUSDT", month="2025-01")

    def test_checksum_mismatch_fails_closed(self):
        source = _fixture_sources()
        url, _ = _archive_url("spot", "BTCUSDT", "2026-02")
        source[url] += b"corruption"
        with self.assertRaisesRegex(ArchiveIntegrityError, "SHA-256 mismatch"):
            _load(source)

    def test_one_missing_bar_fails_strict_grid(self):
        def alter(rows, bars, _funding):
            rows["spot"] = BAR_HEADER + "".join(
                _bar_row(t, spot=True) for t in bars if t != bars[5]
            )

        with self.assertRaisesRegex(ArchiveIntegrityError, "incomplete 4h grid"):
            _load(_fixture_sources(alter=alter))

    def test_zero_daily_quote_volume_is_data_failure(self):
        def alter(rows, bars, _funding):
            rows["spot"] = BAR_HEADER + "".join(
                _bar_row(t, spot=True, quote=0 if t.date() == bars[0].date() else 1000,
                         taker=0 if t.date() == bars[0].date() else 600)
                for t in bars
            )

        with self.assertRaisesRegex(ArchiveIntegrityError, "zero daily quote volume"):
            _load(_fixture_sources(alter=alter))

    def test_taker_buy_quote_cannot_exceed_total_quote(self):
        moment = pd.Timestamp("2026-02-01", tz="UTC")
        with self.assertRaisesRegex(ArchiveIntegrityError, "exceeds quote volume"):
            _parse_bars(_bar_row(moment, spot=True, taker=1001).encode(),
                        kind="spot", symbol="BTCUSDT", month="2026-02")

    def test_unsorted_or_duplicate_bars_and_mixed_units_fail(self):
        a = pd.Timestamp("2026-02-01", tz="UTC")
        b = a + pd.Timedelta(hours=4)
        with self.assertRaisesRegex(ArchiveIntegrityError, "nonmonotone"):
            _parse_bars((_bar_row(b, spot=True) + _bar_row(a, spot=True)).encode(),
                        kind="spot", symbol="BTCUSDT", month="2026-02")
        with self.assertRaisesRegex(ArchiveIntegrityError, "duplicate"):
            _parse_bars((_bar_row(a, spot=True) * 2).encode(),
                        kind="spot", symbol="BTCUSDT", month="2026-02")
        with self.assertRaisesRegex(ArchiveIntegrityError, "expected microseconds"):
            _parse_bars((_bar_row(a, spot=True) + _bar_row(b)).encode(),
                        kind="spot", symbol="BTCUSDT", month="2026-02")

    def test_invalid_ohlc_and_negative_volume_fail(self):
        moment = pd.Timestamp("2026-02-01", tz="UTC")
        with self.assertRaisesRegex(ArchiveIntegrityError, "inconsistent OHLC"):
            _parse_bars(_bar_row(moment, spot=True, high=90).encode(),
                        kind="spot", symbol="BTCUSDT", month="2026-02")
        with self.assertRaisesRegex(ArchiveIntegrityError, "negative volume"):
            _parse_bars(_bar_row(moment, spot=True, volume=-1).encode(),
                        kind="spot", symbol="BTCUSDT", month="2026-02")

    def test_funding_gap_over_24h_fails(self):
        def alter(rows, _bars, funding):
            rows["funding"] = FUNDING_HEADER + "".join(
                _funding_row(t) for t in funding if t not in funding[3:7]
            )

        with self.assertRaisesRegex(ArchiveIntegrityError, "gap exceeds 24 hours"):
            _load(_fixture_sources(alter=alter))

    def test_unknown_symbols_are_outside_frozen_cohort(self):
        with self.assertRaisesRegex(ValueError, "outside frozen study"):
            load_and_audit(symbols=("SOLUSDT",), start_month="2026-02",
                           end_month="2026-02", fetcher={}.__getitem__)


if __name__ == "__main__":
    unittest.main()
