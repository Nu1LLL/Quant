"""Verified Binance BTCUSDT daily spot bars for the difficulty experiment.

Only official monthly ZIPs and their adjacent CHECKSUM files are accepted.
Archive bytes and CSV text stay in memory.  A missing or malformed bar is a
data failure; this module never fills prices or calculates trading results.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import math
import re
import time
from typing import Callable
from zipfile import BadZipFile, ZipFile

import requests


UTC = timezone.utc
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d"
START = datetime(2017, 12, 1, tzinfo=UTC)
LAST_OPEN = datetime(2026, 10, 1, tzinfo=UTC)
END_EXCLUSIVE = LAST_OPEN + timedelta(days=1)
MAX_ZIP_BYTES = 8 * 1024 * 1024
MAX_CSV_BYTES = 4 * 1024 * 1024
MAX_CHECKSUM_BYTES = 256
BAR_HEADER = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
)


class ArchiveIntegrityError(ValueError):
    """The official data cannot be verified or has an invalid schema/grid."""


class ArchiveUnavailable(ArchiveIntegrityError):
    """An official archive could not be obtained; no alternative is inferred."""


@dataclass(frozen=True)
class Bar:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    quote_volume: float

    @property
    def date(self) -> date:
        return self.open_time.date()


@dataclass(frozen=True)
class MonthManifest:
    month: str
    url: str
    sha256: str
    zip_bytes: int
    rows: int


@dataclass(frozen=True)
class MonthArchive:
    bars: tuple[Bar, ...]
    manifest: MonthManifest


@dataclass(frozen=True)
class BarsResult:
    bars: tuple[Bar, ...]
    manifest: tuple[MonthManifest, ...]


def archive_url(month: str) -> tuple[str, str]:
    """Return the frozen official monthly ZIP URL and exact member basename."""
    if not re.fullmatch(r"20\d{2}-(?:0[1-9]|1[0-2])", month):
        raise ValueError("month must be YYYY-MM in 2000-2099")
    basename = f"BTCUSDT-1d-{month}.zip"
    return f"{BASE_URL}/{basename}", basename


def _http_fetch(url: str) -> bytes:
    """Download into bounded memory; retry only transient HTTP/network errors."""
    cap = MAX_CHECKSUM_BYTES if url.endswith(".CHECKSUM") else MAX_ZIP_BYTES
    headers = {"User-Agent": "MiniMedallion-BTCDifficultyArchiveAuditor/1.0"}
    for attempt in range(4):
        try:
            with requests.get(url, headers=headers, timeout=(10, 60), stream=True) as response:
                if response.status_code == 404:
                    raise ArchiveUnavailable(f"official archive unavailable (404): {url}")
                if response.status_code in (408, 429, 500, 502, 503, 504) and attempt < 3:
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
        except (requests.Timeout, requests.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as exc:
            if attempt == 3:
                raise ArchiveUnavailable(f"download failed after retries: {url}") from exc
            time.sleep(0.5 * 2**attempt)
        except requests.RequestException as exc:
            raise ArchiveUnavailable(f"official archive HTTP error: {url}") from exc
    raise RuntimeError("unreachable retry state")


def _checksum_digest(payload: bytes, basename: str) -> str:
    try:
        line = payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ArchiveIntegrityError(f"{basename}: non-ASCII CHECKSUM") from exc
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+(\*?\S+)", line)
    if match is None or match.group(2).removeprefix("*") != basename:
        raise ArchiveIntegrityError(f"{basename}: invalid CHECKSUM")
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
    except (BadZipFile, EOFError, OSError, RuntimeError) as exc:
        raise ArchiveIntegrityError(f"{basename}: invalid ZIP or CRC") from exc
    try:
        return csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ArchiveIntegrityError(f"{basename}: non-UTF-8 CSV") from exc


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    year, number = map(int, month.split("-"))
    start = datetime(year, number, 1, tzinfo=UTC)
    end = (datetime(year + 1, 1, 1, tzinfo=UTC) if number == 12 else
           datetime(year, number + 1, 1, tzinfo=UTC))
    return start, end


def _timestamp(raw: str, label: str, expected_digits: int) -> datetime:
    if re.fullmatch(r"[0-9]+", raw) is None or len(raw) != expected_digits:
        unit = "milliseconds" if expected_digits == 13 else "microseconds"
        raise ArchiveIntegrityError(f"{label}: expected {unit} timestamp")
    microseconds = int(raw) * (1000 if expected_digits == 13 else 1)
    try:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=microseconds)
    except OverflowError as exc:
        raise ArchiveIntegrityError(f"{label}: timestamp out of range") from exc


def _number(raw: str, label: str, *, positive: bool = False) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ArchiveIntegrityError(f"{label}: non-numeric value") from exc
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ArchiveIntegrityError(f"{label}: invalid value")
    return value


def _parse_bars(csv_text: str, month: str, basename: str) -> tuple[Bar, ...]:
    try:
        rows = list(csv.reader(StringIO(csv_text, newline=""), strict=True))
    except csv.Error as exc:
        raise ArchiveIntegrityError(f"{basename}: malformed CSV") from exc
    if not rows or any(not row for row in rows):
        raise ArchiveIntegrityError(f"{basename}: empty or blank CSV row")
    if rows[0] == list(BAR_HEADER):
        rows.pop(0)
    elif rows[0][0] in (BAR_HEADER[0], "\ufeff" + BAR_HEADER[0]):
        raise ArchiveIntegrityError(f"{basename}: unexpected CSV header")
    if not rows or any(len(row) != 12 for row in rows):
        raise ArchiveIntegrityError(f"{basename}: expected 12 columns and data rows")

    start, end = _month_bounds(month)
    expected_digits = 13 if month < "2025-01" else 16
    tick = timedelta(milliseconds=1) if expected_digits == 13 else timedelta(microseconds=1)
    period = timedelta(days=1)
    bars: list[Bar] = []
    previous_open: datetime | None = None
    for line_number, row in enumerate(rows, 1):
        where = f"{basename} row {line_number}"
        opened = _timestamp(row[0], f"{where} open_time", expected_digits)
        closed = _timestamp(row[6], f"{where} close_time", expected_digits)
        if not start <= opened < end or opened.time() != datetime.min.time():
            raise ArchiveIntegrityError(f"{where}: off UTC daily grid or archive month")
        if closed != opened + period - tick:
            raise ArchiveIntegrityError(f"{where}: invalid or early close_time")
        if previous_open is not None and opened <= previous_open:
            raise ArchiveIntegrityError(f"{where}: duplicate or unsorted bar time")
        previous_open = opened

        opened_price = _number(row[1], f"{where} open", positive=True)
        high = _number(row[2], f"{where} high", positive=True)
        low = _number(row[3], f"{where} low", positive=True)
        close = _number(row[4], f"{where} close", positive=True)
        if low > min(opened_price, close) or high < max(opened_price, close) or low > high:
            raise ArchiveIntegrityError(f"{where}: inconsistent OHLC")
        base_volume = _number(row[5], f"{where} volume")
        quote_volume = _number(row[7], f"{where} quote_volume")
        if (base_volume == 0) != (quote_volume == 0):
            raise ArchiveIntegrityError(f"{where}: inconsistent base/quote volume")
        if re.fullmatch(r"[0-9]+", row[8]) is None:
            raise ArchiveIntegrityError(f"{where}: invalid trade count")
        _number(row[9], f"{where} taker_buy_volume")
        _number(row[10], f"{where} taker_buy_quote_volume")
        _number(row[11], f"{where} ignore")
        bars.append(Bar(opened, opened_price, high, low, close, quote_volume))
    return tuple(bars)


def fetch_and_parse_month(
    month: str, *, fetcher: Callable[[str], bytes] | None = None
) -> MonthArchive:
    """Fetch, verify, and parse one BTCUSDT spot 1d monthly archive."""
    url, basename = archive_url(month)
    get_bytes = fetcher or _http_fetch
    try:
        checksum_payload = get_bytes(url + ".CHECKSUM")
        if len(checksum_payload) > MAX_CHECKSUM_BYTES:
            raise ArchiveIntegrityError(f"{basename}: oversized CHECKSUM")
        expected_sha = _checksum_digest(checksum_payload, basename)
        zip_payload = get_bytes(url)
    except ArchiveIntegrityError:
        raise
    except (OSError, requests.RequestException) as exc:
        raise ArchiveUnavailable(f"download failed: {url}") from exc
    if len(zip_payload) > MAX_ZIP_BYTES:
        raise ArchiveIntegrityError(f"{basename}: oversized ZIP")
    actual_sha = sha256(zip_payload).hexdigest()
    if actual_sha != expected_sha:
        raise ArchiveIntegrityError(f"{basename}: SHA-256 mismatch")
    bars = _parse_bars(_csv_from_zip(zip_payload, basename), month, basename)
    return MonthArchive(bars, MonthManifest(month, url, actual_sha, len(zip_payload), len(bars)))


def _months() -> tuple[str, ...]:
    months: list[str] = []
    year, number = START.year, START.month
    while (year, number) <= (LAST_OPEN.year, LAST_OPEN.month):
        months.append(f"{year:04d}-{number:02d}")
        year, number = (year + 1, 1) if number == 12 else (year, number + 1)
    return tuple(months)


def fetch_bars(*, fetcher: Callable[[str], bytes] | None = None) -> BarsResult:
    """Load the frozen daily UTC grid, including only the October 2026 opening bar.

    Every required calendar day must have exactly one verified bar.  A failure
    raises ArchiveIntegrityError/ArchiveUnavailable before any result is returned.
    """
    bars: list[Bar] = []
    manifest: list[MonthManifest] = []
    expected_open = START
    for month in _months():
        archive = fetch_and_parse_month(month, fetcher=fetcher)
        manifest.append(archive.manifest)
        for bar in archive.bars:
            if bar.open_time >= END_EXCLUSIVE:
                continue
            if bar.open_time != expected_open:
                raise ArchiveIntegrityError(
                    f"BTCUSDT 1d grid gap/duplicate: expected {expected_open.isoformat()}, "
                    f"received {bar.open_time.isoformat()}"
                )
            bars.append(bar)
            expected_open += timedelta(days=1)
    if expected_open != END_EXCLUSIVE:
        raise ArchiveIntegrityError(
            f"BTCUSDT 1d grid incomplete: expected bar at {expected_open.isoformat()}"
        )
    return BarsResult(tuple(bars), tuple(manifest))
