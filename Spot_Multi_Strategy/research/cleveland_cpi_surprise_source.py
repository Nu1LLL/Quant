"""Pinned Cleveland Fed nowcast vintages and official Binance spot hours."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import csv
import json
import math
import re
import zipfile

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure, _fetch
from research.ww_cpi_disinflation_source import release_manifest, EVENT_MANIFEST


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "preregistrations/2026-11-08_cleveland_cpi_surprise_btc_spot_1h_manifest.csv"
NOWCAST_URL = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json?sc_lang=en"
NOWCAST_SHA = "9253b45d839c5fe1e5fd1e2d583f9f7aa48bedeb02b4a4060a04041d598c17cd"
PRICE_START = pd.Timestamp("2020-12-01", tz="UTC")
PRICE_END = pd.Timestamp("2025-12-31 23:00", tz="UTC")


def _number(point: dict, label: str) -> float:
    if not isinstance(point, dict) or set(point) != {"value", "tooltext"}:
        raise SourceFailure(f"{label}: nowcast point schema changed")
    raw = point["value"]
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise SourceFailure(f"{label}: nowcast value type invalid")
    try:
        value = float(raw)
    except ValueError as exc:
        raise SourceFailure(f"{label}: nowcast value malformed") from exc
    if not math.isfinite(value) or abs(value) > 100:
        raise SourceFailure(f"{label}: nowcast value out of range")
    return value


def parse_nowcasts(body: bytes, releases: tuple[dict, ...]) -> tuple[dict, ...]:
    if sha256(body).hexdigest() != NOWCAST_SHA or len(body) != 7683463:
        raise SourceFailure("Cleveland Fed historical nowcast JSON changed")
    try:
        charts = json.loads(body)
        if not isinstance(charts, list) or len(charts) != 161:
            raise SourceFailure("Cleveland Fed monthly chart count changed")
        indexed: dict[date, dict] = {}
        for chart in charts:
            year, month = map(int, chart["chart"]["subcaption"].split("-"))
            key = date(year, month, 1)
            if key in indexed:
                raise SourceFailure("duplicate Cleveland Fed reference month")
            indexed[key] = chart
        if min(indexed) != date(2013, 7, 1) or max(indexed) != date(2026, 11, 1):
            raise SourceFailure("Cleveland Fed historical month range changed")
        out = []
        for release in releases:
            ts = release["released_at"]
            if not 2021 <= ts.year <= 2025:
                continue
            month = release["reference_month"]
            chart = indexed[month]
            if chart["chart"]["yaxisname"] != "Month-over-month percent change":
                raise SourceFailure("Cleveland Fed monthly measure changed")
            datasets = chart["dataset"]
            if (len(datasets) != 8 or datasets[0]["seriesname"] != "CPI Inflation"
                    or datasets[4]["seriesname"] != "Actual CPI Inflation"):
                raise SourceFailure("Cleveland Fed headline series changed")
            labels = chart["categories"][0]["category"]
            labels = [p["label"] for p in labels if not p.get("vline")]
            forecast_data, actual_data = datasets[0]["data"], datasets[4]["data"]
            if len(labels) != len(forecast_data) or len(labels) != len(actual_data):
                raise SourceFailure("Cleveland Fed chart date/data alignment changed")
            dates = [date(month.year + (int(label[:2]) < month.month),
                          int(label[:2]), int(label[3:])) for label in labels]
            if dates != sorted(set(dates)):
                raise SourceFailure("Cleveland Fed chart dates duplicate/unsorted")
            actual_idx = [i for i, point in enumerate(actual_data)
                          if point.get("value") not in (None, "", "-")]
            before = [i for i, point in enumerate(forecast_data)
                      if dates[i] < ts.date() and point.get("value") not in (None, "", "-")]
            if (len(actual_idx) != 1 or dates[actual_idx[0]] != ts.date()
                    or not before or before[-1] != actual_idx[0] - 1):
                raise SourceFailure(f"{month}: forecast/first actual publication mismatch")
            forecast = _number(forecast_data[before[-1]], "pre-release headline CPI nowcast")
            actual = _number(actual_data[actual_idx[0]], "first headline CPI actual")
            out.append({"released_at": ts, "reference_month": month,
                        "forecast_date": dates[before[-1]], "forecast": forecast,
                        "actual": actual, "surprise": actual - forecast})
        if len(out) != 59:
            raise SourceFailure("Cleveland Fed/BLS event count changed")
        return tuple(out)
    except SourceFailure:
        raise
    except (KeyError, IndexError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        raise SourceFailure(f"Cleveland Fed historical chart invalid: {exc}") from exc


def _manifest() -> tuple[dict, ...]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("month", "market", "symbol", "interval", "official_zip_sha256"):
            raise SourceFailure("frozen BTC hourly manifest header changed")
        rows = tuple(reader)
    expected = pd.period_range("2020-12", "2025-12", freq="M").astype(str).tolist()
    if len(rows) != 61 or [r["month"] for r in rows] != expected:
        raise SourceFailure("frozen BTC hourly manifest month grid changed")
    for row in rows:
        if (set(row) != {"month", "market", "symbol", "interval", "official_zip_sha256"}
                or (row["market"], row["symbol"], row["interval"]) != ("spot", "BTCUSDT", "1h")
                or not re.fullmatch("[0-9a-f]{64}", row["official_zip_sha256"])):
            raise SourceFailure("frozen BTC hourly manifest row changed")
    return rows


def _load_month(row: dict) -> tuple[str, pd.DataFrame, dict]:
    month, pin = row["month"], row["official_zip_sha256"]
    basename = f"BTCUSDT-1h-{month}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/{basename}"
    sidecar = _fetch(url + ".CHECKSUM").decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-f]{64})\s+\*?" + re.escape(basename), sidecar)
    if match is None or match.group(1) != pin:
        raise SourceFailure(f"{month}: official Binance checksum changed")
    body = _fetch(url)
    if sha256(body).hexdigest() != pin:
        raise SourceFailure(f"{month}: Binance archive hash changed")
    member = f"BTCUSDT-1h-{month}.csv"
    try:
        with zipfile.ZipFile(BytesIO(body)) as archive:
            if archive.namelist() != [member]:
                raise SourceFailure(f"{month}: Binance ZIP member changed")
            raw = archive.read(member).decode("utf-8-sig")
        lines = list(csv.reader(raw.splitlines()))
        first = pd.Timestamp(month + "-01", tz="UTC")
        next_month = first + pd.offsets.MonthBegin(1)
        expected_hours = int((next_month - first) / pd.Timedelta(hours=1))
        if len(lines) != expected_hours:
            raise SourceFailure(f"{month}: Binance hourly row count changed")
        unit = 1_000_000 if month >= "2025-01" else 1_000
        records = []
        for i, fields in enumerate(lines):
            if len(fields) != 12:
                raise SourceFailure(f"{month}: Binance hourly field count changed")
            open_ms = int(fields[0]); close_ms = int(fields[6])
            expected_ms = int((first + pd.Timedelta(hours=i)).timestamp()) * unit
            if open_ms != expected_ms or close_ms != expected_ms + 3600 * unit - 1:
                raise SourceFailure(f"{month}: Binance missing/duplicate/off-grid hour")
            op, hi, lo, cl, quote_vol = map(float, (fields[1], fields[2], fields[3], fields[4], fields[7]))
            if (not all(math.isfinite(v) for v in (op, hi, lo, cl, quote_vol))
                    or lo <= 0 or lo > min(op, cl) or hi < max(op, cl) or quote_vol < 0):
                raise SourceFailure(f"{month}: invalid BTC spot candle")
            records.append((op, hi, lo, cl, quote_vol))
        dates = pd.date_range(first, next_month - pd.Timedelta(hours=1), freq="h")
        frame = pd.DataFrame(records, index=dates,
                             columns=("open", "high", "low", "close", "quote_volume"))
        return month, frame, {"month": month, "sha256": pin, "bytes": len(body), "hours": len(frame)}
    except SourceFailure:
        raise
    except (zipfile.BadZipFile, UnicodeError, csv.Error, ValueError, TypeError,
            IndexError, OverflowError, RuntimeError) as exc:
        raise SourceFailure(f"{month}: Binance archive invalid: {exc}") from exc


def load_sources() -> tuple[tuple[dict, ...], pd.DataFrame, dict]:
    releases = release_manifest(EVENT_MANIFEST.read_bytes())
    body = _fetch(NOWCAST_URL)
    events = parse_nowcasts(body, releases)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, _manifest()))
    parts.sort(key=lambda x: x[0])
    prices = pd.concat([part[1] for part in parts])
    if not prices.index.equals(pd.date_range(PRICE_START, PRICE_END, freq="h")):
        raise SourceFailure("Binance joined BTC spot hour grid invalid")
    audit = {"cleveland_nowcast_sha256": NOWCAST_SHA,
             "cleveland_nowcast_bytes": len(body),
             "official_cpi_events": len(events),
             "binance_spot": [part[2] for part in parts]}
    return events, prices, audit
