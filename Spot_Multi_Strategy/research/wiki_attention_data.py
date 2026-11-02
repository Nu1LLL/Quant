"""Fail-closed, in-memory source loader for preregistered Wikipedia attention.

No signals, returns, or parameter choices are made here. The five article
titles, USD-M contracts and date boundaries are fixed by d72a39c. In
particular, the January 2026 ZIPs are SHA-256 checked in full, but only their
first 00:00 UTC record is opened: future bar high/low/close are not observed.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import csv
import json
import re
import threading
import time
from typing import Callable
from urllib.parse import quote
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import requests

from research.altcoin_reversal_archive import (
    _archive_url,
    _month_range,
    _parse_funding,
    _read_zip_csv,
    _numeric,
    _timestamp_series,
)


PAGE_TITLES = {
    "BTCUSDT": "Bitcoin",
    "ETHUSDT": "Ethereum",
    "DOGEUSDT": "Dogecoin",
    "LTCUSDT": "Litecoin",
    "BCHUSDT": "Bitcoin_Cash",
}
START = pd.Timestamp("2020-12-01", tz="UTC")
TERMINAL = pd.Timestamp("2026-01-01", tz="UTC")
PAGE_END = pd.Timestamp("2025-12-31", tz="UTC")
LAST_ARCHIVE_MONTH = "2025-12"
WIKIMEDIA_BASE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/user"
)
USER_AGENT = "MiniMedallionResearch/1.0 (https://github.com/Nu1LLL/Quant)"
BAR_PERIOD = pd.Timedelta(hours=4)
MAX_FUNDING_CLOCK_OFFSET = pd.Timedelta(milliseconds=100)
MAX_TERMINAL_MEMBER_BYTES = 64 * 1024 * 1024
_LOCAL = threading.local()


class ArchiveIntegrityError(ValueError):
    """A requested official source is unavailable, incomplete or malformed."""


@dataclass(frozen=True)
class DataBundle:
    views: dict[str, pd.Series]
    trade_bars: dict[str, pd.DataFrame]
    mark_bars: dict[str, pd.DataFrame]
    funding_events: dict[str, pd.DataFrame]
    manifest: dict


@dataclass(frozen=True)
class _Payload:
    body: bytes
    status_code: int
    content_type: str | None
    final_url: str


def _page_url(title: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    if title not in PAGE_TITLES.values():
        raise ValueError(f"unregistered article title: {title}")
    return (
        f"{WIKIMEDIA_BASE}/{quote(title, safe='_')}/daily/"
        f"{start:%Y%m%d}/{end:%Y%m%d}"
    )


def _http_fetch(url: str) -> _Payload:
    """Bounded retries; each worker owns its own Session.

    Wikipedia calls are made sequentially by load_and_audit. Binance's ZIP
    and CHECKSUM calls may run across a bounded worker pool.
    """
    session = getattr(_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        _LOCAL.session = session
    for attempt in range(4):
        try:
            response = session.get(url, timeout=(10, 60))
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                retry = response.headers.get("Retry-After", "")
                response.close()
                pause = float(retry) if retry.isdigit() else 0.75 * 2**attempt
                time.sleep(min(pause, 30))
                continue
            response.raise_for_status()
            payload = _Payload(
                body=response.content,
                status_code=response.status_code,
                content_type=response.headers.get("Content-Type"),
                final_url=response.url,
            )
            response.close()
            return payload
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 3:
                raise
            time.sleep(0.75 * 2**attempt)
    raise RuntimeError("unreachable HTTP retry state")


def _fetch(fetcher: Callable[[str], bytes | _Payload], url: str) -> _Payload:
    try:
        payload = fetcher(url)
    except (requests.RequestException, OSError, KeyError) as exc:
        raise ArchiveIntegrityError(f"source download failed: {url}: {exc}") from exc
    if isinstance(payload, _Payload):
        return payload
    if isinstance(payload, bytes):
        return _Payload(payload, 200, None, url)
    raise TypeError("fetcher must return bytes or _Payload")


def _pageviews(
    body: bytes, *, title: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    try:
        document = json.loads(body)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveIntegrityError(f"{title}: malformed pageview JSON") from exc
    if not isinstance(document, dict) or not isinstance(document.get("items"), list):
        raise ArchiveIntegrityError(f"{title}: missing pageview items")
    observations: dict[pd.Timestamp, int] = {}
    for item in document["items"]:
        if not isinstance(item, dict):
            raise ArchiveIntegrityError(f"{title}: malformed pageview item")
        if (
            item.get("project") != "en.wikipedia"
            or item.get("article") != title
            or item.get("granularity") != "daily"
            or item.get("access") != "all-access"
            or item.get("agent") != "user"
        ):
            raise ArchiveIntegrityError(f"{title}: unexpected pageview dimensions")
        raw_date = item.get("timestamp")
        if not isinstance(raw_date, str) or not re.fullmatch(r"\d{8}00", raw_date):
            raise ArchiveIntegrityError(f"{title}: malformed daily timestamp")
        try:
            day = pd.to_datetime(raw_date[:8], format="%Y%m%d", utc=True)
        except ValueError as exc:
            raise ArchiveIntegrityError(f"{title}: impossible daily timestamp") from exc
        value = item.get("views")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ArchiveIntegrityError(f"{title}: pageviews must be positive integers")
        if day in observations:
            raise ArchiveIntegrityError(f"{title}: duplicate daily timestamp {day.date()}")
        observations[day] = value
    expected = pd.date_range(start, end, freq="D", tz="UTC", name="date")
    actual = pd.DatetimeIndex(observations, tz="UTC", name="date").sort_values()
    if not actual.equals(expected):
        missing = expected.difference(actual)
        outside = actual.difference(expected)
        raise ArchiveIntegrityError(
            f"{title}: incomplete daily grid (missing={len(missing)}, outside={len(outside)})"
        )
    return pd.Series([observations[day] for day in actual], index=actual,
                     name=title, dtype="int64")


def _load_page(
    title: str, start: pd.Timestamp, end: pd.Timestamp,
    fetcher: Callable[[str], bytes | _Payload],
) -> tuple[pd.Series, dict]:
    url = _page_url(title, start, end)
    response = _fetch(fetcher, url)
    views = _pageviews(response.body, title=title, start=start, end=end)
    detail = {
        "url": url,
        "http_status": response.status_code,
        "http_content_type": response.content_type,
        "http_final_url": response.final_url,
        "response_bytes": len(response.body),
        "sha256": sha256(response.body).hexdigest(),
        "rows": len(views),
        "first": views.index[0].isoformat(),
        "last": views.index[-1].isoformat(),
    }
    return views, detail


def _archive_member(archive_bytes: bytes, basename: str):
    """Return a ZIP handle and its sole safe CSV member; caller closes handle."""
    expected_csv = basename[:-4] + ".csv"
    try:
        archive = ZipFile(BytesIO(archive_bytes))
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != expected_csv:
            raise ArchiveIntegrityError(f"{basename}: unexpected ZIP member")
        member = members[0]
        if member.is_dir() or member.file_size > MAX_TERMINAL_MEMBER_BYTES:
            raise ArchiveIntegrityError(f"{basename}: unsafe ZIP member")
        return archive, member
    except BadZipFile as exc:
        raise ArchiveIntegrityError(f"{basename}: malformed ZIP") from exc


def _terminal_first_row(archive_bytes: bytes, basename: str, kind: str) -> tuple[list[str], str | None]:
    archive, member = _archive_member(archive_bytes, basename)
    try:
        with archive, archive.open(member) as stream:
            first = stream.readline(2048).decode("utf-8-sig").rstrip("\r\n")
            expected_header = (
                "calc_time,funding_interval_hours,last_funding_rate"
                if kind == "fundingRate" else
                "open_time,open,high,low,close,volume,close_time,quote_volume,"
                "count,taker_buy_volume,taker_buy_quote_volume,ignore"
            )
            if first == expected_header:
                header = first
                line = stream.readline(2048).decode("utf-8-sig").rstrip("\r\n")
            else:
                header = None
                line = first
    except (UnicodeError, RuntimeError) as exc:
        raise ArchiveIntegrityError(f"{basename}: invalid terminal CSV") from exc
    try:
        fields = next(csv.reader([line]))
    except csv.Error as exc:
        raise ArchiveIntegrityError(f"{basename}: malformed terminal row") from exc
    if len(fields) != (3 if kind == "fundingRate" else 12):
        raise ArchiveIntegrityError(f"{basename}: terminal row has wrong field count")
    return fields, header


def _terminal_frame(archive_bytes: bytes, basename: str, kind: str,
                    terminal: pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
    fields, header = _terminal_first_row(archive_bytes, basename, kind)
    timestamp = _timestamp_series(pd.Series([fields[0]]), f"{basename} terminal time").iloc[0]
    if kind == "fundingRate":
        if abs(timestamp - terminal) > MAX_FUNDING_CLOCK_OFFSET:
            raise ArchiveIntegrityError(f"{basename}: terminal funding off 4h grid")
        try:
            interval, rate = float(fields[1]), float(fields[2])
        except ValueError as exc:
            raise ArchiveIntegrityError(f"{basename}: invalid terminal funding") from exc
        if not np.isfinite(interval) or interval <= 0 or interval > 24 or not np.isfinite(rate):
            raise ArchiveIntegrityError(f"{basename}: invalid terminal funding")
        frame = pd.DataFrame({
            "funding_time": [terminal], "raw_funding_time": [timestamp],
            "rate": [rate],
            "funding_interval_hours": [interval],
        })
        return frame, header
    if timestamp != terminal:
        raise ArchiveIntegrityError(f"{basename}: missing exact terminal 00:00 bar")
    try:
        opening = float(fields[1])
    except ValueError as exc:
        raise ArchiveIntegrityError(f"{basename}: invalid terminal opening") from exc
    if not np.isfinite(opening) or opening <= 0:
        raise ArchiveIntegrityError(f"{basename}: invalid terminal opening")
    columns = {"open": [opening], "high": [np.nan], "low": [np.nan],
               "close": [np.nan]}
    if kind == "klines":
        columns["quote_volume"] = [np.nan]
    return pd.DataFrame(columns, index=pd.DatetimeIndex([terminal], name="open_time")), header


def _csv_header(csv_bytes: bytes, kind: str) -> str | None:
    first = csv_bytes.splitlines()[0].decode("utf-8-sig")
    if first.startswith("open_time,") or first.startswith("calc_time,"):
        return first
    return None


def _canonicalize_funding(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Map observed millisecond settlement jitter to a 4h mark open.

    Binance's `calc_time` may be a few milliseconds after the nominal
    settlement hour. We preserve it separately and reject offsets above the
    preregistered 100 ms operational tolerance rather than silently rounding.
    """
    raw = pd.DatetimeIndex(frame["funding_time"])
    canonical = raw.round("4h")
    offsets = abs(raw - canonical)
    if (offsets > MAX_FUNDING_CLOCK_OFFSET).any():
        raise ArchiveIntegrityError(f"{label}: funding timestamp off 4h grid")
    result = frame.copy()
    result.insert(1, "raw_funding_time", raw)
    result["funding_time"] = canonical
    if result["funding_time"].duplicated().any():
        raise ArchiveIntegrityError(f"{label}: duplicate canonical funding time")
    return result


def _parse_complete_bars(csv_bytes: bytes, *, kind: str, symbol: str,
                         month: str) -> pd.DataFrame:
    """Parse entire month while honoring Binance ms/us close-time precision."""
    label = f"{kind}/{symbol}/{month}"
    try:
        raw = pd.read_csv(BytesIO(csv_bytes), header=None, dtype=str,
                          encoding="utf-8-sig", keep_default_na=False)
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ArchiveIntegrityError(f"{label}: invalid CSV") from exc
    if raw.empty or raw.shape[1] != 12:
        raise ArchiveIntegrityError(f"{label}: expected 12-column bars")
    expected_header = (
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
        "ignore",
    )
    if raw.iloc[0, 0] == "open_time":
        if tuple(raw.iloc[0].tolist()) != expected_header:
            raise ArchiveIntegrityError(f"{label}: unknown CSV header")
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.empty:
        raise ArchiveIntegrityError(f"{label}: no bars")
    opened = _timestamp_series(raw.iloc[:, 0], f"{label} open_time")
    closed = _timestamp_series(raw.iloc[:, 6], f"{label} close_time")
    close_units = raw.iloc[:, 6].astype(str).str.len().unique()
    if len(close_units) != 1:
        raise ArchiveIntegrityError(f"{label}: mixed close-time units")
    final_tick = pd.Timedelta(microseconds=1) if close_units[0] == 16 else pd.Timedelta(milliseconds=1)
    if not (closed == opened + BAR_PERIOD - final_tick).all():
        raise ArchiveIntegrityError(f"{label}: inconsistent bar close times")
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    if ((opened < month_start) | (opened >= month_end)).any():
        raise ArchiveIntegrityError(f"{label}: bars outside archive month")
    if (
        (opened.dt.hour % 4 != 0).any()
        or (opened.dt.minute != 0).any()
        or (opened.dt.second != 0).any()
        or (opened.dt.microsecond != 0).any()
    ):
        raise ArchiveIntegrityError(f"{label}: bars off UTC 4h grid")
    columns = {
        name: _numeric(raw.iloc[:, index], f"{label} {name}").to_numpy()
        for index, name in ((1, "open"), (2, "high"), (3, "low"), (4, "close"))
    }
    if kind == "klines":
        columns["quote_volume"] = _numeric(
            raw.iloc[:, 7], f"{label} quote_volume"
        ).to_numpy()
    bars = pd.DataFrame(columns, index=pd.DatetimeIndex(opened, name="open_time"))
    if bars.index.has_duplicates:
        raise ArchiveIntegrityError(f"{label}: duplicate bar timestamps")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise ArchiveIntegrityError(f"{label}: non-positive OHLC")
    if (bars["high"] < bars[["open", "close", "low"]].max(axis=1)).any():
        raise ArchiveIntegrityError(f"{label}: high below another OHLC value")
    if (bars["low"] > bars[["open", "close", "high"]].min(axis=1)).any():
        raise ArchiveIntegrityError(f"{label}: low above another OHLC value")
    if kind == "klines" and (bars["quote_volume"] < 0).any():
        raise ArchiveIntegrityError(f"{label}: negative quote volume")
    return bars.sort_index()


def _load_archive(
    kind: str, symbol: str, month: str, terminal_month: bool,
    fetcher: Callable[[str], bytes | _Payload], terminal: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    url, basename = _archive_url(kind, symbol, month)
    checksum_response = _fetch(fetcher, url + ".CHECKSUM")
    try:
        checksum = checksum_response.body.decode("ascii").strip()
    except UnicodeError as exc:
        raise ArchiveIntegrityError(f"{basename}: invalid checksum encoding") from exc
    matched = re.fullmatch(r"([0-9a-fA-F]{64})\s+([^\s]+)", checksum)
    if matched is None or matched.group(2) != basename:
        raise ArchiveIntegrityError(f"{basename}: malformed official checksum")
    response = _fetch(fetcher, url)
    digest = sha256(response.body).hexdigest()
    if digest.lower() != matched.group(1).lower():
        raise ArchiveIntegrityError(f"{basename}: SHA-256 mismatch")
    if terminal_month:
        frame, header = _terminal_frame(response.body, basename, kind, terminal)
        rows_read = 1
    else:
        try:
            csv_bytes = _read_zip_csv(response.body, basename)
            header = _csv_header(csv_bytes, kind)
            if kind == "fundingRate":
                frame = _canonicalize_funding(
                    _parse_funding(csv_bytes, symbol=symbol, month=month),
                    f"{kind}/{symbol}/{month}",
                )
            else:
                frame = _parse_complete_bars(csv_bytes, kind=kind, symbol=symbol,
                                             month=month)
        except ValueError as exc:
            raise ArchiveIntegrityError(f"{basename}: {exc}") from exc
        rows_read = len(frame)
    first = frame["funding_time"].iloc[0] if kind == "fundingRate" else frame.index[0]
    last = frame["funding_time"].iloc[-1] if kind == "fundingRate" else frame.index[-1]
    detail = {
        "url": url,
        "checksum_url": url + ".CHECKSUM",
        "sha256": digest,
        "checksum_sha256": sha256(checksum_response.body).hexdigest(),
        "zip_bytes": len(response.body),
        "member": basename[:-4] + ".csv",
        "header": header,
        "rows_read": rows_read,
        "terminal_only": terminal_month,
        "first": first.isoformat(),
        "last": last.isoformat(),
    }
    if kind == "fundingRate":
        raw = pd.DatetimeIndex(frame["raw_funding_time"])
        canonical = pd.DatetimeIndex(frame["funding_time"])
        detail["raw_first"] = raw[0].isoformat()
        detail["raw_last"] = raw[-1].isoformat()
        detail["max_clock_offset_ms"] = float(
            (abs(raw - canonical) / pd.Timedelta(milliseconds=1)).max()
        )
    return frame, detail


def _audit_bars(frames: list[pd.DataFrame], *, kind: str, symbol: str,
                start: pd.Timestamp, terminal: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    bars = pd.concat(frames).sort_index()
    if bars.index.has_duplicates:
        raise ArchiveIntegrityError(f"{kind}/{symbol}: duplicate bar timestamps")
    expected = pd.date_range(start, terminal, freq=BAR_PERIOD, tz="UTC", name="open_time")
    if not bars.index.equals(expected):
        missing = expected.difference(bars.index)
        outside = bars.index.difference(expected)
        raise ArchiveIntegrityError(
            f"{kind}/{symbol}: incomplete 4h grid (missing={len(missing)}, outside={len(outside)})"
        )
    if bars.loc[terminal, ["high", "low", "close"]].notna().any():
        raise ArchiveIntegrityError(f"{kind}/{symbol}: future terminal bar values exposed")
    return bars, {
        "expected_full_bars": len(expected) - 1,
        "full_bars": len(bars) - 1,
        "terminal_open_only": True,
        "first": bars.index[0].isoformat(),
        "last": bars.index[-1].isoformat(),
    }


def _audit_funding(frames: list[pd.DataFrame], *, symbol: str,
                   start: pd.Timestamp, terminal: pd.Timestamp,
                   mark_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    events = pd.concat(frames, ignore_index=True).sort_values("funding_time")
    events = events.reset_index(drop=True)
    times = pd.DatetimeIndex(events["funding_time"])
    if times.has_duplicates:
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: duplicate funding events")
    if times[0] != start or times[-1] != terminal:
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: missing first/terminal event")
    if not times.isin(mark_index).all():
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: event off 4h mark grid")
    raw_times = pd.DatetimeIndex(events["raw_funding_time"])
    if (abs(raw_times - times) > MAX_FUNDING_CLOCK_OFFSET).any():
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: raw timestamp beyond 100ms tolerance")
    intervals = events["funding_interval_hours"].to_numpy(dtype="float64")
    if not np.isfinite(intervals).all() or (intervals <= 0).any() or (intervals > 24).any():
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: invalid funding interval")
    # pandas may retain microsecond DatetimeIndex storage. Convert explicitly
    # rather than assuming .asi8 is nanoseconds.
    gaps = np.diff(times.to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "h")
    if not np.array_equal(gaps, intervals[1:]):
        raise ArchiveIntegrityError(f"fundingRate/{symbol}: missing or mistimed funding event")
    return events, {
        "events": len(events), "first": times[0].isoformat(),
        "last": times[-1].isoformat(),
        "interval_hours": sorted(float(x) for x in np.unique(intervals)),
    }


def _load_and_audit_impl(
    *, titles: dict[str, str], start: pd.Timestamp, page_end: pd.Timestamp,
    terminal: pd.Timestamp, last_archive_month: str,
    fetcher: Callable[[str], bytes | _Payload], max_workers: int,
) -> DataBundle:
    if max_workers < 1 or max_workers > 12:
        raise ValueError("max_workers must be 1..12")
    if start.tzinfo is None or terminal.tzinfo is None or page_end.tzinfo is None:
        raise ValueError("boundaries must be UTC-aware")
    if start != pd.Timestamp(f"{start:%Y-%m}-01", tz="UTC"):
        raise ValueError("start must be a month start")
    if terminal != pd.Timestamp(f"{terminal:%Y-%m}-01", tz="UTC"):
        raise ValueError("terminal must be a month start")
    if page_end != terminal - pd.Timedelta(days=1):
        raise ValueError("page_end must immediately precede terminal")
    months = _month_range(f"{start:%Y-%m}", last_archive_month)
    if months[-1] != f"{page_end:%Y-%m}":
        raise ValueError("last_archive_month must include page_end")
    views: dict[str, pd.Series] = {}
    wiki_manifest: dict[str, dict] = {}
    # Explicitly sequential: Wikimedia asks clients to finish one request
    # before starting another. No synthetic delays are imposed on fixtures.
    for symbol, title in titles.items():
        views[symbol], wiki_manifest[symbol] = _load_page(
            title, start, page_end, fetcher
        )
    segments: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    archive_manifest: dict[str, dict] = {}
    tasks = [
        (kind, symbol, month, False)
        for symbol in titles for kind in ("klines", "markPriceKlines", "fundingRate")
        for month in months
    ] + [
        (kind, symbol, f"{terminal:%Y-%m}", True)
        for symbol in titles for kind in ("klines", "markPriceKlines", "fundingRate")
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_load_archive, kind, symbol, month, is_terminal,
                        fetcher, terminal): (kind, symbol, month)
            for kind, symbol, month, is_terminal in tasks
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
    for symbol in titles:
        trade_bars[symbol], trade_detail = _audit_bars(
            segments[("klines", symbol)], kind="klines", symbol=symbol,
            start=start, terminal=terminal,
        )
        mark_bars[symbol], mark_detail = _audit_bars(
            segments[("markPriceKlines", symbol)], kind="markPriceKlines",
            symbol=symbol, start=start, terminal=terminal,
        )
        if not trade_bars[symbol].index.equals(mark_bars[symbol].index):
            raise ArchiveIntegrityError(f"{symbol}: trade/mark grids differ")
        funding_events[symbol], funding_detail = _audit_funding(
            segments[("fundingRate", symbol)], symbol=symbol,
            start=start, terminal=terminal, mark_index=mark_bars[symbol].index,
        )
        coverage[symbol] = {
            "trade": trade_detail, "mark": mark_detail,
            "funding": funding_detail,
        }
    manifest = {
        "wiki_source": WIKIMEDIA_BASE,
        "binance_source": "https://data.binance.vision/data/futures/um/monthly",
        "registered_pages": dict(titles),
        "start": start.isoformat(),
        "page_end": page_end.isoformat(),
        "terminal": terminal.isoformat(),
        "wikipedia": wiki_manifest,
        "archive_count": len(archive_manifest),
        "archives": dict(sorted(archive_manifest.items())),
        "coverage": coverage,
    }
    json.dumps(manifest, allow_nan=False)
    return DataBundle(views, trade_bars, mark_bars, funding_events, manifest)


def load_and_audit(
    *, fetcher: Callable[[str], bytes | _Payload] | None = None,
    max_workers: int = 8,
) -> DataBundle:
    """Load the exact five-coin/2021–2025 preregistered source bundle.

    `fetcher` is an injectable URL-to-bytes function for offline tests. Raw
    ZIPs, CSVs and pageview JSON are never written to disk. A source defect
    raises ArchiveIntegrityError; programming errors are not relabelled.
    """
    return _load_and_audit_impl(
        titles=PAGE_TITLES, start=START, page_end=PAGE_END,
        terminal=TERMINAL, last_archive_month=LAST_ARCHIVE_MONTH,
        fetcher=fetcher or _http_fetch, max_workers=max_workers,
    )
