"""Strict, checksum-audited archives for the frozen spot-flow study.

This module only acquires and validates data. It neither calculates a signal
nor reads strategy returns. ZIP contents remain in memory and are not cached.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import re
from typing import Callable

import numpy as np
import pandas as pd
import requests

from .altcoin_reversal_archive import (
    ArchiveIntegrityError,
    _archive_url as _futures_archive_url,
    _http_fetch,
    _month_range,
    _numeric,
    _read_zip_csv,
    _timestamp_series,
)


FIXED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT")
KINDS = ("spot", "trade", "mark", "funding")
SPOT_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
FUTURES_BASE_URL = "https://data.binance.vision/data/futures/um/monthly"
BAR_PERIOD = pd.Timedelta(hours=4)
MAX_FUNDING_GAP = pd.Timedelta(hours=24)
BAR_HEADER = (
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
)
# The existing archive parser accepts this generic kline CSV header. The
# mark-price API documents columns 5 and 7–11 as ignored values; if actual
# archive headers differ, the first formal archive audit fails closed.
FUNDING_HEADER = ("calc_time", "funding_interval_hours", "last_funding_rate")


@dataclass(frozen=True)
class ArchiveBundle:
    spot_bars: dict[str, pd.DataFrame]
    trade_bars: dict[str, pd.DataFrame]
    mark_bars: dict[str, pd.DataFrame]
    funding_events: dict[str, pd.DataFrame]
    manifest: dict


def _archive_url(kind: str, symbol: str, month: str) -> tuple[str, str]:
    if kind not in KINDS:
        raise ValueError(f"unknown archive kind: {kind}")
    if symbol not in FIXED_SYMBOLS:
        raise ValueError(f"symbol outside frozen study: {symbol}")
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise ValueError(f"invalid archive month: {month}")
    if kind == "spot":
        basename = f"{symbol}-4h-{month}.zip"
        return f"{SPOT_BASE_URL}/{symbol}/4h/{basename}", basename
    futures_kind = {"trade": "klines", "mark": "markPriceKlines",
                    "funding": "fundingRate"}[kind]
    return _futures_archive_url(futures_kind, symbol, month)


def _read_csv(csv_bytes: bytes, label: str, columns: int,
              expected_header: tuple[str, ...]) -> pd.DataFrame:
    try:
        raw = pd.read_csv(BytesIO(csv_bytes), header=None, dtype=str,
                          encoding="utf-8-sig", keep_default_na=False)
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ArchiveIntegrityError(f"{label}: invalid CSV") from exc
    if raw.empty or raw.shape[1] != columns:
        raise ArchiveIntegrityError(f"{label}: expected {columns}-column CSV")
    if raw.iloc[0, 0] == expected_header[0]:
        if tuple(raw.iloc[0].tolist()) != expected_header:
            raise ArchiveIntegrityError(f"{label}: unknown CSV header")
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.empty:
        raise ArchiveIntegrityError(f"{label}: no data rows")
    return raw


def _parse_bars(csv_bytes: bytes, *, kind: str, symbol: str,
                month: str) -> pd.DataFrame:
    if kind not in ("spot", "trade", "mark"):
        raise ValueError(f"invalid bar kind: {kind}")
    label = f"{kind}/{symbol}/{month}"
    raw = _read_csv(csv_bytes, label, 12, BAR_HEADER)
    open_text = raw.iloc[:, 0].astype(str).str.strip()
    close_text = raw.iloc[:, 6].astype(str).str.strip()
    # Binance's published archive contract changes *spot* to microseconds on
    # 2025-01-01. USD-M futures remain milliseconds throughout this window.
    expected_digits = 16 if kind == "spot" and month >= "2025-01" else 13
    if (not open_text.str.fullmatch(rf"\d{{{expected_digits}}}").all()
            or not close_text.str.fullmatch(rf"\d{{{expected_digits}}}").all()):
        unit = "microseconds" if expected_digits == 16 else "milliseconds"
        raise ArchiveIntegrityError(f"{label}: expected {unit} timestamps")
    opened = _timestamp_series(raw.iloc[:, 0], f"{label} open_time")
    closed = _timestamp_series(raw.iloc[:, 6], f"{label} close_time")
    if not (open_text.str.len() == close_text.str.len()).all():
        raise ArchiveIntegrityError(f"{label}: mixed open/close timestamp units")
    tick = pd.Timedelta(microseconds=1) if len(open_text.iloc[0]) == 16 else pd.Timedelta(milliseconds=1)
    if not (closed == opened + BAR_PERIOD - tick).all():
        raise ArchiveIntegrityError(f"{label}: inconsistent bar close times")
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    if ((opened < month_start) | (opened >= month_end)).any():
        raise ArchiveIntegrityError(f"{label}: bars outside archive month")
    if not (opened.dt.floor("4h") == opened).all():
        raise ArchiveIntegrityError(f"{label}: bars off UTC 4h grid")
    index = pd.DatetimeIndex(opened, name="open_time")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ArchiveIntegrityError(f"{label}: duplicate or nonmonotone timestamps")

    numeric = {
        name: _numeric(raw.iloc[:, column], f"{label} {name}").to_numpy()
        for column, name in (
            (1, "open"), (2, "high"), (3, "low"), (4, "close"),
        )
    }
    if kind == "mark":
        # The official mark-price REST response is a 12-tuple, but fields
        # 5 and 7–11 are explicitly "Ignore", not traded volume or flow.
        for column in (5, 7, 8, 9, 10, 11):
            _numeric(raw.iloc[:, column], f"{label} ignored_field_{column}")
    else:
        numeric.update({
            name: _numeric(raw.iloc[:, column], f"{label} {name}").to_numpy()
            for column, name in (
                (5, "volume"), (7, "quote_volume"), (8, "count"),
                (9, "taker_buy_volume"), (10, "taker_buy_quote_volume"),
            )
        })
        _numeric(raw.iloc[:, 11], f"{label} ignore")
    bars = pd.DataFrame(numeric, index=index)
    prices = bars[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ArchiveIntegrityError(f"{label}: non-positive OHLC")
    if (bars["high"] < prices.max(axis=1)).any() or (bars["low"] > prices.min(axis=1)).any():
        raise ArchiveIntegrityError(f"{label}: inconsistent OHLC")
    if kind != "mark":
        if (bars[["volume", "quote_volume", "count", "taker_buy_volume",
                  "taker_buy_quote_volume"]] < 0).any().any():
            raise ArchiveIntegrityError(f"{label}: negative volume or count")
        if (bars["count"] != np.floor(bars["count"])).any():
            raise ArchiveIntegrityError(f"{label}: non-integer trade count")
    if kind == "spot":
        if (bars["taker_buy_quote_volume"] > bars["quote_volume"]).any():
            raise ArchiveIntegrityError(f"{label}: taker buy quote volume exceeds quote volume")
        return bars[["open", "high", "low", "close", "quote_volume",
                     "taker_buy_quote_volume"]]
    if kind == "trade":
        return bars[["open", "high", "low", "close", "quote_volume"]]
    return bars[["open", "high", "low", "close"]]


def _parse_funding(csv_bytes: bytes, *, symbol: str,
                   month: str) -> pd.DataFrame:
    label = f"funding/{symbol}/{month}"
    raw = _read_csv(csv_bytes, label, 3, FUNDING_HEADER)
    if not raw.iloc[:, 0].astype(str).str.strip().str.fullmatch(r"\d{13}").all():
        raise ArchiveIntegrityError(f"{label}: expected milliseconds timestamps")
    times = _timestamp_series(raw.iloc[:, 0], f"{label} calc_time")
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    if ((times < month_start) | (times >= month_end)).any():
        raise ArchiveIntegrityError(f"{label}: events outside archive month")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ArchiveIntegrityError(f"{label}: duplicate or nonmonotone funding timestamps")
    intervals = _numeric(raw.iloc[:, 1], f"{label} funding_interval_hours")
    rates = _numeric(raw.iloc[:, 2], f"{label} rate")
    if (intervals <= 0).any() or (intervals > 24).any():
        raise ArchiveIntegrityError(f"{label}: invalid funding interval")
    return pd.DataFrame({"funding_time": times, "rate": rates.to_numpy(),
                         "funding_interval_hours": intervals.to_numpy()})


def _fetch_parse_one(kind: str, symbol: str, month: str,
                     fetcher: Callable[[str], bytes]) -> tuple[pd.DataFrame, dict]:
    url, basename = _archive_url(kind, symbol, month)
    label = f"{kind}/{symbol}/{month}"
    try:
        checksum_payload = fetcher(url + ".CHECKSUM")
        checksum_text = checksum_payload.decode("ascii").strip()
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+([^\s]+)", checksum_text)
        if match is None or match.group(2) != basename:
            raise ArchiveIntegrityError(f"{label}: invalid official checksum")
        archive_bytes = fetcher(url)
    except (requests.RequestException, OSError, KeyError, UnicodeError) as exc:
        raise ArchiveIntegrityError(f"{label}: download failed: {exc}") from exc
    digest = sha256(archive_bytes).hexdigest()
    if digest.lower() != match.group(1).lower():
        raise ArchiveIntegrityError(f"{label}: SHA-256 mismatch")
    size = len(archive_bytes)
    csv_bytes = _read_zip_csv(archive_bytes, basename)
    del archive_bytes
    if kind == "funding":
        frame = _parse_funding(csv_bytes, symbol=symbol, month=month)
    else:
        frame = _parse_bars(csv_bytes, kind=kind, symbol=symbol, month=month)
    return frame, {"sha256": digest, "rows": int(len(frame)), "zip_bytes": size}


def _audit_bars(frames: list[pd.DataFrame], *, kind: str, symbol: str,
                start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    bars = pd.concat(frames)
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ArchiveIntegrityError(f"{kind}/{symbol}: duplicate or nonmonotone bars across months")
    expected = pd.date_range(start, end - BAR_PERIOD, freq=BAR_PERIOD,
                             tz="UTC", name="open_time")
    if not bars.index.equals(expected):
        missing = expected.difference(bars.index)
        extra = bars.index.difference(expected)
        raise ArchiveIntegrityError(
            f"{kind}/{symbol}: incomplete 4h grid; missing={len(missing)}, extra={len(extra)}"
        )
    if kind == "spot":
        quote_by_day = bars["quote_volume"].resample("1D").sum()
        if (quote_by_day <= 0).any():
            raise ArchiveIntegrityError(f"spot/{symbol}: zero daily quote volume")
    return bars, {"expected_bars": int(len(expected)),
                  "actual_bars": int(len(bars)), "coverage": 1.0,
                  "first": bars.index[0].isoformat(),
                  "last": bars.index[-1].isoformat()}


def _audit_funding(frames: list[pd.DataFrame], *, symbol: str,
                   start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    events = pd.concat(frames, ignore_index=True)
    times = events["funding_time"]
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ArchiveIntegrityError(f"funding/{symbol}: duplicate or nonmonotone events")
    if times.iloc[0] < start or times.iloc[-1] >= end:
        raise ArchiveIntegrityError(f"funding/{symbol}: events outside study period")
    if times.iloc[0] - start > MAX_FUNDING_GAP or end - times.iloc[-1] > MAX_FUNDING_GAP:
        raise ArchiveIntegrityError(f"funding/{symbol}: missing first or last 24 hours")
    gaps = times.diff().dropna()
    if (gaps > MAX_FUNDING_GAP).any():
        raise ArchiveIntegrityError(f"funding/{symbol}: gap exceeds 24 hours")
    return events, {"events": int(len(events)),
                    "first": times.iloc[0].isoformat(),
                    "last": times.iloc[-1].isoformat(),
                    "max_gap_hours": float(gaps.max() / pd.Timedelta(hours=1)) if len(gaps) else 0.0,
                    "funding_intervals_hours": sorted(float(value) for value in events["funding_interval_hours"].unique())}


def load_and_audit(
    *,
    symbols: tuple[str, ...] = FIXED_SYMBOLS,
    start_month: str = "2021-01",
    end_month: str = "2026-09",
    max_workers: int = 12,
    fetcher: Callable[[str], bytes] | None = None,
) -> ArchiveBundle:
    """Validate every frozen monthly source, failing the whole study on one defect.

    The optional byte fetcher exists solely for deterministic offline tests.
    Raw archives are never persisted by this module.
    """
    months = _month_range(start_month, end_month)
    symbols = tuple(symbols)
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("symbols must be nonempty and unique")
    if any(symbol not in FIXED_SYMBOLS for symbol in symbols):
        raise ValueError("symbol outside frozen study")
    if max_workers < 1 or max_workers > 32:
        raise ValueError("max_workers must be between 1 and 32")
    start = pd.Timestamp(f"{months[0]}-01", tz="UTC")
    end = pd.Timestamp(f"{months[-1]}-01", tz="UTC") + pd.offsets.MonthBegin(1)
    get_bytes = fetcher or _http_fetch
    tasks = ((kind, symbol, month) for symbol in symbols
             for kind in KINDS for month in months)
    segments: dict[tuple[str, str, str], pd.DataFrame] = {}
    archive_manifest: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_parse_one, kind, symbol, month, get_bytes):
            (kind, symbol, month)
            for kind, symbol, month in tasks
        }
        for future in as_completed(futures):
            kind, symbol, month = futures[future]
            try:
                frame, detail = future.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
            segments[(kind, symbol, month)] = frame
            archive_manifest[f"{kind}/{symbol}/{month}"] = detail
    by_kind = {kind: {} for kind in KINDS}
    coverage = {}
    for symbol in symbols:
        coverage[symbol] = {}
        for kind in KINDS:
            frames = [segments[(kind, symbol, month)] for month in months]
            if kind == "funding":
                checked, status = _audit_funding(frames, symbol=symbol,
                                                 start=start, end=end)
            else:
                checked, status = _audit_bars(frames, kind=kind,
                                              symbol=symbol, start=start, end=end)
            by_kind[kind][symbol] = checked
            coverage[symbol][kind] = status
    manifest = {
        "sources": {"spot": SPOT_BASE_URL, "futures": FUTURES_BASE_URL},
        "symbols": list(symbols), "start_month": start_month,
        "end_month": end_month, "archive_count": len(archive_manifest),
        "archives": dict(sorted(archive_manifest.items())),
        "coverage": coverage,
    }
    json.dumps(manifest, allow_nan=False)
    return ArchiveBundle(by_kind["spot"], by_kind["trade"], by_kind["mark"],
                         by_kind["funding"], manifest)
