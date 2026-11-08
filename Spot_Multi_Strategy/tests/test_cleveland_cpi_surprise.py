"""Synthetic timing, cost, and archive-integrity checks without reading results."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from unittest.mock import patch
import unittest
import csv
import zipfile

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure
from research.cleveland_cpi_surprise_source import _load_month
from research.cleveland_cpi_surprise_sim import _schedule, simulate


def _events():
    start = datetime(2021, 1, 14, 13, 30, tzinfo=timezone.utc)
    return tuple({"released_at": start + timedelta(days=30 * i),
                  "surprise": -0.1 if i == 0 else 0.1}
                 for i in range(59))


def test_release_timing_uses_next_hour_and_seven_days():
    schedule, entries = _schedule(_events(), 0, False)
    assert entries == 1
    assert schedule == {pd.Timestamp("2021-01-14 14:00", tz="UTC"): "buy",
                        pd.Timestamp("2021-01-21 14:00", tz="UTC"): "sell"}
    delayed, _ = _schedule(_events(), 1, False)
    assert pd.Timestamp("2021-01-14 15:00", tz="UTC") in delayed


def test_flat_price_account_loses_only_frozen_costs():
    index = pd.date_range("2020-12-01", "2025-12-31 23:00", tz="UTC", freq="h")
    prices = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                           "close": 100.0, "quote_volume": 2_000_000.0}, index=index)
    net = simulate(_events(), prices)
    gross = simulate(_events(), prices, slippage=0, fee_rate=0)
    hold = simulate(_events(), prices, buy_and_hold=True)
    assert len(net.days) == 1826
    assert len(net.fills) == 2
    assert net.terminal_cash < 10_000
    assert abs(gross.terminal_cash - 10_000) < 1e-6
    assert len(hold.fills) == 2 and hold.terminal_cash < 10_000
    assert net.days[-1].end == net.terminal_cash


def _archive(month: str, unit: int, missing_last: bool = False) -> bytes:
    first = pd.Timestamp(month + "-01", tz="UTC")
    next_month = first + pd.offsets.MonthBegin(1)
    count = int((next_month - first) / pd.Timedelta(hours=1)) - int(missing_last)
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        rows = []
        for i in range(count):
            stamp = int((first + pd.Timedelta(hours=i)).timestamp()) * unit
            rows.append([stamp, 100, 100, 100, 100, 1, stamp + 3600 * unit - 1,
                         1_000_000, 1, 1, 1_000_000, 0])
        from io import StringIO
        buffer = StringIO()
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        archive.writestr(f"BTCUSDT-1h-{month}.csv", buffer.getvalue())
    return stream.getvalue()


def test_hourly_archive_validates_timestamp_unit():
    for month, unit in (("2020-12", 1000), ("2025-01", 1_000_000)):
        body = _archive(month, unit)
        digest = sha256(body).hexdigest()
        name = f"BTCUSDT-1h-{month}.zip"
        with patch("research.cleveland_cpi_surprise_source._fetch",
                   side_effect=[f"{digest}  {name}\n".encode(), body]):
            _, frame, audit = _load_month({"month": month, "official_zip_sha256": digest})
        assert len(frame) in (744, 720)
        assert audit["sha256"] == digest


def test_archive_gap_rejects_instead_of_shortening_sample():
    month = "2025-01"
    body = _archive(month, 1_000_000, missing_last=True)
    digest = sha256(body).hexdigest()
    with patch("research.cleveland_cpi_surprise_source._fetch",
               side_effect=[f"{digest}  BTCUSDT-1h-{month}.zip\n".encode(), body]):
        try:
            _load_month({"month": month, "official_zip_sha256": digest})
        except SourceFailure as exc:
            assert "row count" in str(exc)
        else:
            raise AssertionError("missing last hour was accepted")


class ClevelandSyntheticTests(unittest.TestCase):
    def test_release_timing(self):
        test_release_timing_uses_next_hour_and_seven_days()

    def test_flat_account(self):
        test_flat_price_account_loses_only_frozen_costs()

    def test_archive_units(self):
        test_hourly_archive_validates_timestamp_unit()

    def test_archive_gap(self):
        test_archive_gap_rejects_instead_of_shortening_sample()


if __name__ == "__main__":
    unittest.main()
