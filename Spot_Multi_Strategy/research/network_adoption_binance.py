"""Fail-closed, in-memory Binance archives for the network-adoption experiment.

This module only verifies and parses official monthly archives.  It neither
selects a surviving universe nor calculates a signal or trading result.
Callers must provide independently verified listing, suspension and delisting
information when interpreting absent archives or bars.  A 404 is *not* proof
of a delisting.  ZIPs and CSV bytes are never written to disk.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import math
import re
import time
from typing import Callable, Iterable
from zipfile import BadZipFile, ZipFile

import requests


UTC = timezone.utc
FIXED_SYMBOLS = (
    "BATUSDT", "CVCUSDT", "KNCUSDT", "MKRUSDT", "OMGUSDT", "RENUSDT", "ZRXUSDT",
)
BASE_URL = "https://data.binance.vision/data"
KINDS = frozenset(("trade4", "mark4", "mark1", "funding", "btc_spot1d"))
MAX_ZIP_BYTES = 32 * 1024 * 1024
MAX_CSV_BYTES = 64 * 1024 * 1024
MAX_CHECKSUM_BYTES = 256
BAR_HEADER = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
)
FUNDING_HEADER = ("calc_time", "funding_interval_hours", "last_funding_rate")


class ArchiveIntegrityError(ValueError):
    """An archive is missing, unverified, malformed, or internally inconsistent."""


class ArchiveUnavailable(ArchiveIntegrityError):
    """A requested official archive is unavailable; this is not delisting proof."""


@dataclass(frozen=True)
class Bar:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    quote_volume: float | None


@dataclass(frozen=True)
class Funding:
    calc_time: datetime
    rate: float
    interval_hours: float


@dataclass(frozen=True)
class MonthArchive:
    rows: tuple[Bar | Funding, ...]
    sha256: str
    zip_bytes: int
    url: str


def archive_url(kind: str, symbol: str, month: str) -> tuple[str, str]:
    """Return the official monthly ZIP URL and its exact expected basename."""
    if kind not in KINDS:
        raise ValueError(f"unknown archive kind: {kind}")
    if not re.fullmatch(r"20\d{2}-(?:0[1-9]|1[0-2])", month):
        raise ValueError("month must be YYYY-MM in 2000-2099")
    if kind == "btc_spot1d":
        if symbol != "BTCUSDT":
            raise ValueError("btc_spot1d is fixed to BTCUSDT")
        basename = f"BTCUSDT-1d-{month}.zip"
        return (
            f"{BASE_URL}/spot/monthly/klines/BTCUSDT/1d/{basename}", basename
        )
    if symbol not in FIXED_SYMBOLS:
        raise ValueError(f"symbol not in frozen network-adoption universe: {symbol}")
    base = f"{BASE_URL}/futures/um/monthly"
    if kind == "funding":
        basename = f"{symbol}-fundingRate-{month}.zip"
        return f"{base}/fundingRate/{symbol}/{basename}", basename
    archive_kind = "klines" if kind == "trade4" else "markPriceKlines"
    interval = "1h" if kind == "mark1" else "4h"
    basename = f"{symbol}-{interval}-{month}.zip"
    return f"{base}/{archive_kind}/{symbol}/{interval}/{basename}", basename


def _http_fetch(url: str) -> bytes:
    """Bound memory and retry only transient network/server failures."""
    cap = MAX_CHECKSUM_BYTES if url.endswith(".CHECKSUM") else MAX_ZIP_BYTES
    headers = {"User-Agent": "MiniMedallion-NetworkArchiveAuditor/1.0"}
    for attempt in range(4):
        try:
            with requests.get(url, headers=headers, timeout=(10, 60), stream=True) as response:
                if response.status_code == 404:
                    raise ArchiveUnavailable(f"official archive unavailable (404): {url}")
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < 3:
                        time.sleep(0.5 * 2**attempt)
                        continue
                response.raise_for_status()
                chunks = BytesIO()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunks.tell() + len(chunk) > cap:
                        raise ArchiveIntegrityError(f"download exceeds {cap} bytes: {url}")
                    chunks.write(chunk)
                return chunks.getvalue()
        except ArchiveIntegrityError:
            raise
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 3:
                raise ArchiveUnavailable(f"download failed after retries: {url}") from exc
            time.sleep(0.5 * 2**attempt)
        except requests.HTTPError as exc:
            raise ArchiveUnavailable(f"official archive HTTP error: {url}") from exc
    raise RuntimeError("unreachable retry state")


def _checksum_digest(payload: bytes, expected_basename: str) -> str:
    try:
        line = payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ArchiveIntegrityError("non-ASCII checksum") from exc
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+(\*?\S+)", line)
    if match is None or match.group(2).removeprefix("*") != expected_basename:
        raise ArchiveIntegrityError(f"invalid checksum for {expected_basename}")
    return match.group(1).lower()


def _csv_from_zip(payload: bytes, basename: str) -> str:
    expected_csv = basename.removesuffix(".zip") + ".csv"
    try:
        with ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != expected_csv:
                raise ArchiveIntegrityError(
                    f"{basename}: ZIP must contain exactly {expected_csv}"
                )
            member = members[0]
            if member.is_dir() or member.file_size > MAX_CSV_BYTES:
                raise ArchiveIntegrityError(f"{basename}: unsafe ZIP member")
            with archive.open(member) as handle:
                csv_bytes = handle.read(MAX_CSV_BYTES + 1)
            if len(csv_bytes) != member.file_size or len(csv_bytes) > MAX_CSV_BYTES:
                raise ArchiveIntegrityError(f"{basename}: invalid decompressed size")
    except (BadZipFile, EOFError, OSError) as exc:
        raise ArchiveIntegrityError(f"{basename}: invalid ZIP/CRC") from exc
    try:
        return csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ArchiveIntegrityError(f"{basename}: non-UTF-8 CSV") from exc


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    year, month_num = map(int, month.split("-"))
    start = datetime(year, month_num, 1, tzinfo=UTC)
    next_month = datetime(year + 1, 1, 1, tzinfo=UTC) if month_num == 12 else datetime(
        year, month_num + 1, 1, tzinfo=UTC
    )
    return start, next_month


def _timestamp(raw: str, label: str) -> tuple[datetime, str]:
    if not re.fullmatch(r"\d{13}|\d{16}", raw):
        raise ArchiveIntegrityError(f"{label}: unsupported timestamp unit/value")
    unit = "ms" if len(raw) == 13 else "us"
    microseconds = int(raw) * (1000 if unit == "ms" else 1)
    try:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=microseconds), unit
    except OverflowError as exc:
        raise ArchiveIntegrityError(f"{label}: timestamp out of range") from exc


def _number(raw: str, label: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ArchiveIntegrityError(f"{label}: non-numeric value") from exc
    if not math.isfinite(value):
        raise ArchiveIntegrityError(f"{label}: non-finite value")
    if positive and value <= 0:
        raise ArchiveIntegrityError(f"{label}: must be positive")
    if nonnegative and value < 0:
        raise ArchiveIntegrityError(f"{label}: must be nonnegative")
    return value


def _rows(csv_text: str, expected_width: int, header: tuple[str, ...], label: str) -> list[list[str]]:
    try:
        rows = list(csv.reader(StringIO(csv_text, newline=""), strict=True))
    except csv.Error as exc:
        raise ArchiveIntegrityError(f"{label}: malformed CSV") from exc
    if not rows:
        raise ArchiveIntegrityError(f"{label}: empty CSV")
    if any(not row for row in rows):
        raise ArchiveIntegrityError(f"{label}: blank CSV row")
    if rows[0] == list(header):
        rows.pop(0)
    elif rows[0][0] in (header[0], "\ufeff" + header[0]):
        raise ArchiveIntegrityError(f"{label}: unexpected CSV header")
    if not rows or any(len(row) != expected_width for row in rows):
        raise ArchiveIntegrityError(f"{label}: expected {expected_width} columns and data rows")
    return rows


def _parse_bars(csv_text: str, kind: str, month: str, label: str) -> tuple[Bar, ...]:
    rows = _rows(csv_text, 12, BAR_HEADER, label)
    start, end = _month_bounds(month)
    period = timedelta(days=1) if kind == "btc_spot1d" else timedelta(
        hours=1 if kind == "mark1" else 4
    )
    parsed: list[Bar] = []
    previous_time: datetime | None = None
    for line_number, row in enumerate(rows, 1):
        where = f"{label} row {line_number}"
        opened, unit = _timestamp(row[0], f"{where} open_time")
        closed, close_unit = _timestamp(row[6], f"{where} close_time")
        if not start <= opened < end:
            raise ArchiveIntegrityError(f"{where}: outside archive month")
        if (opened - datetime(1970, 1, 1, tzinfo=UTC)) % period != timedelta(0):
            raise ArchiveIntegrityError(f"{where}: off UTC bar grid")
        if close_unit != unit:
            raise ArchiveIntegrityError(f"{where}: mixed timestamp units")
        close_tick = timedelta(milliseconds=1) if unit == "ms" else timedelta(microseconds=1)
        if closed != opened + period - close_tick:
            raise ArchiveIntegrityError(f"{where}: invalid close_time")
        if previous_time is not None and opened <= previous_time:
            raise ArchiveIntegrityError(f"{where}: duplicate or unsorted bar time")
        previous_time = opened
        o = _number(row[1], f"{where} open", positive=True)
        h = _number(row[2], f"{where} high", positive=True)
        l = _number(row[3], f"{where} low", positive=True)
        c = _number(row[4], f"{where} close", positive=True)
        if h < max(o, l, c) or l > min(o, h, c):
            raise ArchiveIntegrityError(f"{where}: inconsistent OHLC")
        _number(row[5], f"{where} volume", nonnegative=True)
        quote = _number(row[7], f"{where} quote_volume", nonnegative=True)
        if not re.fullmatch(r"\d+", row[8]):
            raise ArchiveIntegrityError(f"{where}: invalid trade count")
        _number(row[9], f"{where} taker_buy_volume", nonnegative=True)
        _number(row[10], f"{where} taker_buy_quote_volume", nonnegative=True)
        _number(row[11], f"{where} ignore")
        parsed.append(Bar(opened, o, h, l, c,
                          quote if kind in ("trade4", "btc_spot1d") else None))
    return tuple(parsed)


def _parse_funding(csv_text: str, month: str, label: str) -> tuple[Funding, ...]:
    rows = _rows(csv_text, 3, FUNDING_HEADER, label)
    start, end = _month_bounds(month)
    parsed: list[Funding] = []
    previous_time: datetime | None = None
    for line_number, row in enumerate(rows, 1):
        where = f"{label} row {line_number}"
        calc_time, _ = _timestamp(row[0], f"{where} calc_time")
        if not start <= calc_time < end:
            raise ArchiveIntegrityError(f"{where}: outside archive month")
        # Binance's official calc_time occasionally has millisecond jitter.
        # Preserve it exactly; the frozen experiment rejects a non-whole-hour
        # event only if a position was active at that instant.  Rounding here
        # would silently change the cashflow timing and violate the prereg.
        if previous_time is not None and calc_time <= previous_time:
            raise ArchiveIntegrityError(f"{where}: duplicate or unsorted funding time")
        previous_time = calc_time
        interval = _number(row[1], f"{where} funding_interval_hours", positive=True)
        if interval > 24:
            raise ArchiveIntegrityError(f"{where}: funding interval exceeds 24 hours")
        rate = _number(row[2], f"{where} last_funding_rate")
        parsed.append(Funding(calc_time, rate, interval))
    return tuple(parsed)


def fetch_and_parse_month(
    kind: str,
    symbol: str,
    month: str,
    *,
    fetcher: Callable[[str], bytes] | None = None,
) -> MonthArchive:
    """Fetch, SHA-256 verify and parse exactly one official monthly archive.

    Missing archives raise ArchiveUnavailable.  This function does not infer
    whether an absent month reflects delisting, outage or data loss.
    """
    url, basename = archive_url(kind, symbol, month)
    get_bytes = fetcher or _http_fetch
    try:
        checksum_payload = get_bytes(url + ".CHECKSUM")
        if len(checksum_payload) > MAX_CHECKSUM_BYTES:
            raise ArchiveIntegrityError(f"oversized checksum: {basename}")
        expected_sha = _checksum_digest(checksum_payload, basename)
        zip_payload = get_bytes(url)
    except ArchiveIntegrityError:
        raise
    except (OSError, requests.RequestException) as exc:
        raise ArchiveUnavailable(f"download failed: {url}") from exc
    if len(zip_payload) > MAX_ZIP_BYTES:
        raise ArchiveIntegrityError(f"oversized ZIP: {basename}")
    zip_size = len(zip_payload)
    actual_sha = sha256(zip_payload).hexdigest()
    if actual_sha != expected_sha:
        raise ArchiveIntegrityError(f"SHA-256 mismatch: {basename}")
    csv_text = _csv_from_zip(zip_payload, basename)
    del zip_payload
    label = f"{kind}/{symbol}/{month}"
    rows = (_parse_funding(csv_text, month, label) if kind == "funding"
            else _parse_bars(csv_text, kind, month, label))
    return MonthArchive(rows, actual_sha, zip_size, url)


def load_month(
    kind: str,
    symbol: str,
    month: str,
    *,
    fetcher: Callable[[str], bytes] | None = None,
) -> tuple[Bar | Funding, ...]:
    """Return parsed bars/events only; use fetch_and_parse_month for a digest manifest."""
    return fetch_and_parse_month(kind, symbol, month, fetcher=fetcher).rows


def bar_grid_gaps(
    bars: Iterable[Bar], start: datetime, end: datetime, *, kind: str
) -> tuple[datetime, ...]:
    """List absent bar opens in [start,end), without excusing any absence.

    Caller must independently verify every exception (e.g. an official halt).
    Also rejects duplicate, non-grid, or out-of-window observations.
    """
    if kind not in ("trade4", "mark4", "mark1", "btc_spot1d"):
        raise ValueError("bar_grid_gaps requires a bar kind")
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("start/end must be aware and increasing")
    period = timedelta(days=1) if kind == "btc_spot1d" else timedelta(
        hours=1 if kind == "mark1" else 4
    )
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    if (start - epoch) % period or (end - epoch) % period:
        raise ValueError("start/end must be on requested UTC bar grid")
    observed: set[datetime] = set()
    for bar in bars:
        stamp = bar.open_time
        if not start <= stamp < end or (stamp - epoch) % period:
            raise ArchiveIntegrityError(f"{kind}: bar outside requested time grid: {stamp}")
        if stamp in observed:
            raise ArchiveIntegrityError(f"{kind}: duplicate bar: {stamp}")
        observed.add(stamp)
    missing: list[datetime] = []
    cursor = start
    while cursor < end:
        if cursor not in observed:
            missing.append(cursor)
        cursor += period
    return tuple(missing)
