"""Audited Binance USD-M monthly archives for the preregistered reversal test.

This module does not calculate signals or returns. Every requested monthly ZIP
must match the corresponding official .CHECKSUM before its CSV is parsed. ZIPs
are held only in memory and are never extracted to the filesystem.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import re
import threading
import time
from typing import Callable
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import requests


FIXED_SYMBOLS = (
    "AAVEUSDT", "BANDUSDT", "COMPUSDT", "DASHUSDT", "IOTAUSDT",
    "KSMUSDT", "ONTUSDT", "SNXUSDT", "SUSHIUSDT", "XMRUSDT",
    "XTZUSDT", "YFIUSDT", "ZECUSDT", "ZILUSDT",
)
LISTING_START = {
    # The official opening was 07:00 UTC. The raw 04:00 trading bar is only
    # partially open; 08:00 is the first complete four-hour trading bar.
    "AAVEUSDT": pd.Timestamp("2020-10-16 08:00:00", tz="UTC"),
    "KSMUSDT": pd.Timestamp("2020-10-14 08:00:00", tz="UTC"),
}
KINDS = ("klines", "markPriceKlines", "fundingRate")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly"
BAR_PERIOD = pd.Timedelta(hours=4)
MAX_UNCOMPRESSED_CSV_BYTES = 64 * 1024 * 1024
_LOCAL = threading.local()


class ArchiveIntegrityError(ValueError):
    """Official archive is unavailable, corrupted, incomplete, or malformed."""


@dataclass(frozen=True)
class ArchiveBundle:
    trade_bars: dict[str, pd.DataFrame]
    mark_bars: dict[str, pd.DataFrame]
    funding_events: dict[str, pd.DataFrame]
    manifest: dict


def _month_range(start_month: str, end_month: str) -> tuple[str, ...]:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", start_month):
        raise ValueError("start_month must be YYYY-MM")
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", end_month):
        raise ValueError("end_month must be YYYY-MM")
    start, end = pd.Period(start_month, freq="M"), pd.Period(end_month, freq="M")
    if start > end:
        raise ValueError("start_month must not exceed end_month")
    return tuple(str(month) for month in pd.period_range(start, end, freq="M"))


def _archive_url(kind: str, symbol: str, month: str) -> tuple[str, str]:
    if kind not in KINDS:
        raise ValueError(f"unknown archive kind: {kind}")
    if not re.fullmatch(r"[A-Z0-9]+USDT", symbol):
        raise ValueError(f"invalid USD-M USDT symbol: {symbol}")
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise ValueError(f"invalid archive month: {month}")
    if kind == "fundingRate":
        basename = f"{symbol}-fundingRate-{month}.zip"
        return f"{BASE_URL}/fundingRate/{symbol}/{basename}", basename
    basename = f"{symbol}-4h-{month}.zip"
    return f"{BASE_URL}/{kind}/{symbol}/4h/{basename}", basename


def _http_fetch(url: str) -> bytes:
    """Thread-local HTTP pooling with bounded retries for transient failures."""
    session = getattr(_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "MiniMedallionArchiveAuditor/1.0"})
        _LOCAL.session = session
    for attempt in range(4):
        try:
            response = session.get(url, timeout=(10, 60))
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                response.close()
                time.sleep(0.75 * (2 ** attempt))
                continue
            response.raise_for_status()
            content = response.content
            response.close()
            return content
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 3:
                raise
            time.sleep(0.75 * (2 ** attempt))
    raise RuntimeError("unreachable HTTP retry state")


def _read_zip_csv(archive_bytes: bytes, expected_basename: str) -> bytes:
    expected_csv_name = expected_basename[:-4] + ".csv"
    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != expected_csv_name:
                raise ArchiveIntegrityError(
                    f"{expected_basename}: ZIP must contain exactly {expected_csv_name}"
                )
            member = members[0]
            if member.is_dir() or member.file_size > MAX_UNCOMPRESSED_CSV_BYTES:
                raise ArchiveIntegrityError(f"{expected_basename}: unsafe ZIP member")
            with archive.open(member) as handle:
                payload = handle.read(MAX_UNCOMPRESSED_CSV_BYTES + 1)
            if len(payload) != member.file_size or len(payload) > MAX_UNCOMPRESSED_CSV_BYTES:
                raise ArchiveIntegrityError(f"{expected_basename}: invalid decompressed size")
            return payload
    except BadZipFile as exc:
        raise ArchiveIntegrityError(f"{expected_basename}: invalid ZIP") from exc


def _timestamp_series(values: pd.Series, label: str) -> pd.Series:
    raw = values.astype(str).str.strip()
    if raw.empty or not raw.str.fullmatch(r"\d{13}|\d{16}").all():
        raise ArchiveIntegrityError(f"{label}: non-integer or unsupported timestamp")
    lengths = raw.str.len().unique()
    if len(lengths) != 1:
        raise ArchiveIntegrityError(f"{label}: mixed millisecond/microsecond timestamp units")
    unit = "ms" if lengths[0] == 13 else "us"
    try:
        return pd.to_datetime(raw.astype("int64"), unit=unit, utc=True)
    except (ValueError, OverflowError) as exc:
        raise ArchiveIntegrityError(f"{label}: timestamp out of range") from exc


def _numeric(values: pd.Series, label: str) -> pd.Series:
    try:
        out = pd.to_numeric(values, errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise ArchiveIntegrityError(f"{label}: non-numeric value") from exc
    if not np.isfinite(out.to_numpy()).all():
        raise ArchiveIntegrityError(f"{label}: NaN or infinite value")
    return out


def _parse_bars(csv_bytes: bytes, *, kind: str, symbol: str, month: str) -> pd.DataFrame:
    try:
        raw = pd.read_csv(BytesIO(csv_bytes), header=None, dtype=str,
                          encoding="utf-8-sig", keep_default_na=False)
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ArchiveIntegrityError(f"{kind}/{symbol}/{month}: invalid CSV") from exc
    if raw.shape[1] != 12 or raw.empty:
        raise ArchiveIntegrityError(f"{kind}/{symbol}/{month}: expected 12-column bars")
    if raw.iloc[0, 0] == "open_time":
        expected_header = (
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore",
        )
        if tuple(raw.iloc[0].tolist()) != expected_header:
            raise ArchiveIntegrityError(f"{kind}/{symbol}/{month}: unknown CSV header")
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.empty:
        raise ArchiveIntegrityError(f"{kind}/{symbol}/{month}: no bars")
    label = f"{kind}/{symbol}/{month}"
    timestamp = _timestamp_series(raw.iloc[:, 0], f"{label} open_time")
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    if ((timestamp < month_start) | (timestamp >= month_end)).any():
        raise ArchiveIntegrityError(f"{label}: bars outside archive month")
    if (timestamp.dt.hour % 4 != 0).any() or (timestamp.dt.minute != 0).any() or (timestamp.dt.second != 0).any() or (timestamp.dt.microsecond != 0).any():
        raise ArchiveIntegrityError(f"{label}: bars off UTC 4h grid")
    close_timestamp = _timestamp_series(raw.iloc[:, 6], f"{label} close_time")
    if not (close_timestamp == timestamp + BAR_PERIOD - pd.Timedelta(milliseconds=1)).all():
        raise ArchiveIntegrityError(f"{label}: inconsistent bar close times")
    columns = {}
    for index, name in ((1, "open"), (2, "high"), (3, "low"), (4, "close")):
        columns[name] = _numeric(raw.iloc[:, index], f"{label} {name}").to_numpy()
    if kind == "klines":
        columns["quote_volume"] = _numeric(raw.iloc[:, 7], f"{label} quote_volume").to_numpy()
    bars = pd.DataFrame(columns, index=pd.DatetimeIndex(timestamp, name="open_time"))
    if bars.index.has_duplicates:
        raise ArchiveIntegrityError(f"{label}: duplicate bar timestamps")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise ArchiveIntegrityError(f"{label}: non-positive OHLC")
    if (bars["high"] < bars[["open", "close", "low"]].max(axis=1)).any() or (bars["low"] > bars[["open", "close", "high"]].min(axis=1)).any():
        raise ArchiveIntegrityError(f"{label}: inconsistent OHLC")
    if "quote_volume" in bars and (bars["quote_volume"] < 0).any():
        raise ArchiveIntegrityError(f"{label}: negative quote volume")
    return bars.sort_index()


def _parse_funding(csv_bytes: bytes, *, symbol: str, month: str) -> pd.DataFrame:
    label = f"fundingRate/{symbol}/{month}"
    try:
        raw = pd.read_csv(BytesIO(csv_bytes), header=None, dtype=str,
                          encoding="utf-8-sig", keep_default_na=False)
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ArchiveIntegrityError(f"{label}: invalid CSV") from exc
    if raw.shape[1] != 3 or raw.empty:
        raise ArchiveIntegrityError(f"{label}: expected 3-column funding CSV")
    if raw.iloc[0, 0] == "calc_time":
        if tuple(raw.iloc[0].tolist()) != (
            "calc_time", "funding_interval_hours", "last_funding_rate"
        ):
            raise ArchiveIntegrityError(f"{label}: unknown funding header")
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.empty:
        raise ArchiveIntegrityError(f"{label}: no funding events")
    timestamp = _timestamp_series(raw.iloc[:, 0], f"{label} calc_time")
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    if ((timestamp < month_start) | (timestamp >= month_end)).any():
        raise ArchiveIntegrityError(f"{label}: funding outside archive month")
    interval = _numeric(raw.iloc[:, 1], f"{label} funding_interval_hours")
    if (interval <= 0).any() or (interval > 24).any():
        raise ArchiveIntegrityError(f"{label}: invalid funding interval")
    rate = _numeric(raw.iloc[:, 2], f"{label} funding_rate")
    events = pd.DataFrame({
        "funding_time": timestamp,
        "rate": rate.to_numpy(),
        "funding_interval_hours": interval.to_numpy(),
    })
    if events["funding_time"].duplicated().any():
        raise ArchiveIntegrityError(f"{label}: duplicate funding timestamps")
    return events.sort_values("funding_time").reset_index(drop=True)


def _fetch_parse_one(kind: str, symbol: str, month: str,
                     fetcher: Callable[[str], bytes]) -> tuple[pd.DataFrame, dict]:
    url, basename = _archive_url(kind, symbol, month)
    try:
        checksum_payload = fetcher(url + ".CHECKSUM")
        checksum_text = checksum_payload.decode("ascii").strip()
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+([^\s]+)", checksum_text)
        if match is None or match.group(2) != basename:
            raise ArchiveIntegrityError(f"{kind}/{symbol}/{month}: invalid official checksum")
        archive_bytes = fetcher(url)
    except (requests.RequestException, UnicodeError, OSError) as exc:
        raise ArchiveIntegrityError(f"{kind}/{symbol}/{month}: download failed: {exc}") from exc
    actual_sha = sha256(archive_bytes).hexdigest()
    if actual_sha.lower() != match.group(1).lower():
        raise ArchiveIntegrityError(f"{kind}/{symbol}/{month}: SHA-256 mismatch")
    archive_size = len(archive_bytes)
    csv_bytes = _read_zip_csv(archive_bytes, basename)
    del archive_bytes
    if kind == "fundingRate":
        frame = _parse_funding(csv_bytes, symbol=symbol, month=month)
    else:
        frame = _parse_bars(csv_bytes, kind=kind, symbol=symbol, month=month)
    return frame, {"sha256": actual_sha, "rows": int(len(frame)), "zip_bytes": archive_size}


def _audit_bars(frames: list[pd.DataFrame], *, kind: str, symbol: str,
                start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    bars = pd.concat(frames, axis=0).sort_index()
    if bars.index.has_duplicates:
        raise ArchiveIntegrityError(f"{kind}/{symbol}: duplicate timestamps across months")
    # AAVE and KSM opened at 07:00 UTC. Their raw trade file has a partial
    # 04:00 bar and the mark file starts at 00:00. Neither the partial trade
    # bar nor the two pre-08:00 mark bars are complete tradable observations.
    prelisting_count = int((bars.index < start).sum())
    if symbol in LISTING_START and start == LISTING_START[symbol]:
        prelisting = bars.loc[bars.index < start]
        expected_count = 1 if kind == "klines" else 2
        expected_prelisting = pd.date_range(
            start - expected_count * BAR_PERIOD,
            start - BAR_PERIOD,
            freq=BAR_PERIOD,
            tz="UTC",
            name="open_time",
        )
        if not prelisting.index.equals(expected_prelisting):
            raise ArchiveIntegrityError(
                f"{kind}/{symbol}: unexpected partial/prelisting observations"
            )
        bars = bars.loc[bars.index >= start]
    elif prelisting_count:
        raise ArchiveIntegrityError(f"{kind}/{symbol}: bars before tradable start")
    if bars.empty or bars.index[0] != start:
        raise ArchiveIntegrityError(f"{kind}/{symbol}: missing first tradable bar")
    expected = pd.date_range(start, end - BAR_PERIOD, freq=BAR_PERIOD, tz="UTC",
                             name="open_time")
    if not bars.index.isin(expected).all():
        raise ArchiveIntegrityError(f"{kind}/{symbol}: bars outside requested grid")
    missing = expected.difference(bars.index)
    coverage = len(bars) / len(expected)
    if coverage < 0.999:
        raise ArchiveIntegrityError(
            f"{kind}/{symbol}: coverage {coverage:.6f} below 0.999"
        )
    if len(missing):
        runs = pd.Series(missing).diff().ne(BAR_PERIOD).cumsum()
        max_missing_run = int(pd.Series(1, index=missing).groupby(runs.to_numpy()).sum().max())
        if max_missing_run > 1:
            raise ArchiveIntegrityError(
                f"{kind}/{symbol}: missing-bar gap exceeds 4 hours"
            )
    else:
        max_missing_run = 0
    return bars, {
        "expected_bars": int(len(expected)),
        "actual_bars": int(len(bars)),
        "coverage": float(coverage),
        "missing_bars": int(len(missing)),
        "max_missing_run_4h_bars": max_missing_run,
        "prelisting_or_partial_bars_excluded": prelisting_count,
        "first": bars.index[0].isoformat(),
        "last": bars.index[-1].isoformat(),
    }


def _audit_funding(frames: list[pd.DataFrame], *, symbol: str,
                   start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    events = pd.concat(frames, ignore_index=True).sort_values("funding_time")
    events = events.reset_index(drop=True)
    timestamps = events["funding_time"]
    if timestamps.duplicated().any():
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: duplicate timestamps across months")
    if timestamps.iloc[0] < start or timestamps.iloc[-1] >= end:
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: funding outside tradable period")
    gaps = timestamps.diff().dropna()
    if timestamps.iloc[0] - start > pd.Timedelta(hours=24):
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: missing initial funding events")
    if end - timestamps.iloc[-1] > pd.Timedelta(hours=24):
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: missing final funding events")
    if (gaps > pd.Timedelta(hours=24)).any():
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: funding gap exceeds 24 hours")
    return events, {
        "events": int(len(events)),
        "first": timestamps.iloc[0].isoformat(),
        "last": timestamps.iloc[-1].isoformat(),
        "max_gap_hours": float(gaps.max() / pd.Timedelta(hours=1)) if len(gaps) else 0.0,
        "funding_intervals_hours": sorted(
            float(value) for value in events["funding_interval_hours"].unique()
        ),
    }


def load_and_audit(
    *,
    symbols: tuple[str, ...] = FIXED_SYMBOLS,
    start_month: str = "2020-10",
    end_month: str = "2026-09",
    max_workers: int = 12,
    fetcher: Callable[[str], bytes] | None = None,
) -> ArchiveBundle:
    """Download, checksum-verify and fully audit every fixed monthly archive.

    No archive is persisted. `fetcher` is solely an injected byte-fetch function
    for deterministic offline tests. The returned manifest is JSON serializable.
    A data defect raises ArchiveIntegrityError and does not drop a symbol.
    """
    months = _month_range(start_month, end_month)
    symbols = tuple(symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be nonempty and unique")
    for symbol in symbols:
        _archive_url("klines", symbol, months[0])
    if max_workers < 1 or max_workers > 32:
        raise ValueError("max_workers must be between 1 and 32")
    first = pd.Timestamp(f"{months[0]}-01", tz="UTC")
    last = pd.Timestamp(f"{months[-1]}-01", tz="UTC") + pd.offsets.MonthBegin(1)
    get_bytes = fetcher or _http_fetch
    segments: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    archive_manifest: dict[str, dict] = {}
    tasks = ((kind, symbol, month) for symbol in symbols
             for kind in KINDS for month in months)
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
            segments[(kind, symbol)].append(frame)
            archive_manifest[f"{kind}/{symbol}/{month}"] = detail
    trade_bars, mark_bars, funding_events = {}, {}, {}
    coverage = {}
    for symbol in symbols:
        listed = LISTING_START.get(symbol, first)
        effective_start = max(first, listed)
        trade_bars[symbol], trade_status = _audit_bars(
            segments[("klines", symbol)], kind="klines", symbol=symbol,
            start=effective_start, end=last,
        )
        mark_bars[symbol], mark_status = _audit_bars(
            segments[("markPriceKlines", symbol)], kind="markPriceKlines",
            symbol=symbol, start=effective_start, end=last,
        )
        funding_events[symbol], funding_status = _audit_funding(
            segments[("fundingRate", symbol)], symbol=symbol,
            start=effective_start, end=last,
        )
        coverage[symbol] = {
            "trade": trade_status, "mark": mark_status,
            "funding": funding_status,
        }
    manifest = {
        "source": BASE_URL,
        "symbols": list(symbols),
        "start_month": start_month,
        "end_month": end_month,
        "archive_count": len(archive_manifest),
        "archives": dict(sorted(archive_manifest.items())),
        "coverage": coverage,
    }
    json.dumps(manifest, allow_nan=False)
    return ArchiveBundle(trade_bars, mark_bars, funding_events, manifest)
