"""Bounded, read-only audit of Binance Vision monthly 1m/5m klines.

This module emits archive integrity metadata, not prices, signals or returns.
An HTTP fetcher must be passed explicitly. Historical ZIP publication time is
not a claim that a strategy observed each bar at its original close time.
"""

from __future__ import annotations

from calendar import monthrange
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import re
from struct import unpack
from tempfile import SpooledTemporaryFile
from typing import Callable, Iterable, Iterator
from zipfile import BadZipFile, LargeZipFile, ZipFile


MAX_CHECKSUM_BYTES = 512
MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_CSV_BYTES = 64 * 1024 * 1024
MAX_CSV_LINE_BYTES = 4096
MAX_CENTRAL_DIRECTORY_BYTES = 4096
SPOOL_MEMORY_BYTES = 4 * 1024 * 1024
MAX_FETCH_CHUNK_BYTES = 64 * 1024
_INTERVAL_SECONDS = {"1m": 60, "5m": 300}
_SYMBOL = re.compile(r"[A-Z0-9]{3,32}\Z")
_MONTH = re.compile(r"(\d{4})-(0[1-9]|1[0-2])\Z")
_HEADER = (
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
)


class ArchiveIntegrityError(ValueError):
    """A required archive is missing, over budget or internally inconsistent."""


ChunkFetcher = Callable[[str], Iterable[bytes]]


@dataclass(frozen=True)
class ArchiveRequest:
    market: str  # spot or USD-margined futures (um)
    symbol: str
    month: str
    interval: str = "5m"

    def __post_init__(self) -> None:
        if self.market not in {"spot", "um"}:
            raise ValueError("market must be spot or um")
        if not _SYMBOL.fullmatch(self.symbol):
            raise ValueError("symbol must be a canonical uppercase Binance symbol")
        if not _MONTH.fullmatch(self.month):
            raise ValueError("month must be YYYY-MM")
        if int(self.month[:4]) < 1970:
            raise ValueError("month year must be at least 1970")
        if self.interval not in _INTERVAL_SECONDS:
            raise ValueError("interval must be 1m or 5m")


@dataclass(frozen=True)
class VerifiedKline:
    """One bar from a checksum-verified *whole* archive, not a live quote.

    ``window_end`` is exclusive, unlike Binance's inclusive CSV close time.
    The monthly ZIP checksum does not prove when an individual historical bar
    was published or whether its archived value was ever revised.
    """

    request: ArchiveRequest
    open_time: datetime
    window_end: datetime
    quote_volume: Decimal
    taker_buy_quote_volume: Decimal
    trade_count: int
    archive_sha256: str
    source_url: str
    publication_time_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ArchiveRequest):
            raise TypeError("request must be ArchiveRequest")
        if (not isinstance(self.open_time, datetime)
                or not isinstance(self.window_end, datetime)
                or self.open_time.tzinfo is None or self.open_time.utcoffset() != timedelta(0)
                or self.window_end.tzinfo is None or self.window_end.utcoffset() != timedelta(0)):
            raise ValueError("kline times must be UTC")
        if self.window_end - self.open_time != timedelta(seconds=_INTERVAL_SECONDS[self.request.interval]):
            raise ValueError("kline duration disagrees with request")
        if (not isinstance(self.quote_volume, Decimal)
                or not isinstance(self.taker_buy_quote_volume, Decimal)
                or not self.quote_volume.is_finite()
                or not self.taker_buy_quote_volume.is_finite()
                or not 0 <= self.taker_buy_quote_volume <= self.quote_volume):
            raise ValueError("invalid verified kline quote volumes")
        if type(self.trade_count) is not int or self.trade_count < 0:
            raise ValueError("invalid verified kline trade count")
        if not re.fullmatch(r"[0-9a-f]{64}", self.archive_sha256):
            raise ValueError("archive_sha256 must be lowercase SHA-256")
        if self.source_url != archive_url(self.request)[0]:
            raise ValueError("source_url disagrees with request")


def archive_url(request: ArchiveRequest) -> tuple[str, str]:
    """Return the documented monthly kline ZIP URL and exact ZIP basename."""
    if not isinstance(request, ArchiveRequest):
        raise TypeError("request must be ArchiveRequest")
    basename = f"{request.symbol}-{request.interval}-{request.month}.zip"
    prefix = "spot" if request.market == "spot" else "futures/um"
    return (
        f"https://data.binance.vision/data/{prefix}/monthly/klines/"
        f"{request.symbol}/{request.interval}/{basename}",
        basename,
    )


def http_chunk_fetcher(url: str) -> Iterator[bytes]:
    """Opt-in HTTP transport. The auditor never calls it implicitly."""
    import requests

    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            yield from response.iter_content(chunk_size=64 * 1024)
    except requests.RequestException as exc:
        raise ArchiveIntegrityError(f"archive transport failed: {url}") from exc


def _chunks(fetch_chunks: ChunkFetcher, url: str) -> Iterator[bytes]:
    try:
        stream = iter(fetch_chunks(url))
        try:
            for chunk in stream:
                if not isinstance(chunk, bytes):
                    raise ArchiveIntegrityError("transport yielded a non-bytes chunk")
                if len(chunk) > MAX_FETCH_CHUNK_BYTES:
                    raise ArchiveIntegrityError("transport chunk exceeds byte limit")
                if chunk:
                    yield chunk
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
    except (KeyError, OSError) as exc:
        raise ArchiveIntegrityError(f"archive source unavailable: {url}") from exc


def _checksum(fetch_chunks: ChunkFetcher, url: str, basename: str) -> str:
    payload = bytearray()
    for chunk in _chunks(fetch_chunks, url + ".CHECKSUM"):
        if len(payload) + len(chunk) > MAX_CHECKSUM_BYTES:
            raise ArchiveIntegrityError("CHECKSUM response exceeds byte limit")
        payload.extend(chunk)
    try:
        statement = payload.decode("ascii").strip()
    except UnicodeError as exc:
        raise ArchiveIntegrityError("CHECKSUM is not ASCII") from exc
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != basename:
        raise ArchiveIntegrityError("CHECKSUM must name exactly the requested ZIP")
    return match.group(1).lower()


def _month_limits(request: ArchiveRequest) -> tuple[int, int, int]:
    year, month = map(int, request.month.split("-"))
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
    days = monthrange(year, month)[1]
    return start, start + days * 86400, days * 86400 // _INTERVAL_SECONDS[request.interval]


def _decimal(raw: str, label: str, *, positive: bool = False) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ArchiveIntegrityError(f"{label} is not numeric") from exc
    if not value.is_finite() or (value <= 0 if positive else value < 0):
        raise ArchiveIntegrityError(f"{label} is nonfinite or outside its valid range")
    return value


def _audit_csv(member, request: ArchiveRequest) -> tuple[int, int]:
    start_s, end_s, expected_rows = _month_limits(request)
    microseconds = request.market == "spot" and request.month >= "2025-01"
    unit = 1_000_000 if microseconds else 1_000
    digits = 16 if microseconds else 13
    expected_open = start_s * unit
    end_open = end_s * unit
    interval = _INTERVAL_SECONDS[request.interval] * unit
    csv_bytes = rows = line_number = 0

    while True:
        raw_line = member.readline(MAX_CSV_LINE_BYTES + 1)
        if not raw_line:
            break
        if len(raw_line) > MAX_CSV_LINE_BYTES:
            raise ArchiveIntegrityError("CSV line exceeds byte limit")
        line_number += 1
        csv_bytes += len(raw_line)
        if csv_bytes > MAX_CSV_BYTES:
            raise ArchiveIntegrityError("CSV exceeds uncompressed byte limit")
        try:
            line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")
            parsed = next(csv.reader([line], strict=True))
        except (UnicodeError, csv.Error, StopIteration) as exc:
            raise ArchiveIntegrityError("malformed CSV row") from exc
        if tuple(parsed) == _HEADER:
            if line_number == 1:
                continue
            raise ArchiveIntegrityError("repeated CSV header")
        if len(parsed) != 12:
            raise ArchiveIntegrityError("kline CSV must contain exactly 12 columns")
        opened_text, closed_text = parsed[0], parsed[6]
        if (
            len(opened_text) != digits or not opened_text.isascii() or not opened_text.isdigit()
            or len(closed_text) != digits or not closed_text.isascii() or not closed_text.isdigit()
        ):
            raise ArchiveIntegrityError("incorrect kline timestamp unit or format")
        opened = int(opened_text)
        closed = int(closed_text)
        if opened != expected_open or opened >= end_open:
            raise ArchiveIntegrityError("missing, duplicate or off-grid intraday bar")
        if closed != opened + interval - 1:
            raise ArchiveIntegrityError("kline close time is not the interval boundary")
        high = _decimal(parsed[2], "high", positive=True)
        low = _decimal(parsed[3], "low", positive=True)
        opening = _decimal(parsed[1], "open", positive=True)
        closing = _decimal(parsed[4], "close", positive=True)
        if high < max(opening, closing, low) or low > min(opening, closing, high):
            raise ArchiveIntegrityError("inconsistent OHLC")
        volume = _decimal(parsed[5], "base volume")
        quote = _decimal(parsed[7], "quote volume")
        taker_base = _decimal(parsed[9], "taker buy base volume")
        taker_quote = _decimal(parsed[10], "taker buy quote volume")
        if taker_base > volume or taker_quote > quote:
            raise ArchiveIntegrityError("taker buy volume exceeds total volume")
        count_text = parsed[8]
        if not count_text.isascii() or not count_text.isdigit():
            raise ArchiveIntegrityError("trade count must be a nonnegative integer")
        if (int(count_text) == 0) != (quote == 0):
            raise ArchiveIntegrityError("trade count and quote volume disagree")
        _decimal(parsed[11], "ignore field")
        expected_open += interval
        rows += 1

    if rows != expected_rows or expected_open != end_open:
        raise ArchiveIntegrityError("month ends before the complete intraday grid")
    return rows, csv_bytes


def _check_zip_directory(spool, zip_bytes: int) -> None:
    """Bound ZipFile's central-directory allocation before it parses entries.

    Binance's monthly archive is a single, ordinary ZIP member. Reject ZIP64,
    multivolume ZIPs, comments and preambles instead of letting a malformed
    archive expand into a large list of ZipInfo objects.
    """
    if zip_bytes < 22:
        raise ArchiveIntegrityError("invalid ZIP end-of-central-directory record")
    spool.seek(-22, 2)
    footer = spool.read(22)
    signature, disk, start_disk, disk_entries, total_entries, size, offset, comment = unpack(
        "<4sHHHHIIH", footer,
    )
    if (signature != b"PK\x05\x06" or disk != 0 or start_disk != 0
            or disk_entries != 1 or total_entries != 1 or comment != 0
            or size > MAX_CENTRAL_DIRECTORY_BYTES or offset + size != zip_bytes - 22):
        raise ArchiveIntegrityError("ZIP directory must contain one bounded CSV entry")


@contextmanager
def _opened_verified_archive(
    request: ArchiveRequest,
    *,
    fetch_chunks: ChunkFetcher,
    max_zip_bytes: int = MAX_ZIP_BYTES,
) -> Iterator[tuple[ZipFile, object, dict[str, object]]]:
    """Keep one fully verified ZIP open only within the caller's context.

    Compressed archive bytes are held in a temporary spooled file that rolls
    to temporary disk after a 4 MiB threshold, then is deleted on exit. The
    fetcher must yield at most 64 KiB per chunk. The CSV is validated line by
    line before any caller may inspect or emit a row.
    ``max_zip_bytes`` can tighten, but not expand, the 64 MiB hard cap.
    """
    url, basename = archive_url(request)
    if type(max_zip_bytes) is not int or not 1 <= max_zip_bytes <= MAX_ZIP_BYTES:
        raise ValueError("max_zip_bytes must be within the fixed archive limit")
    expected_hash = _checksum(fetch_chunks, url, basename)
    digest = sha256()
    zip_bytes = 0
    with SpooledTemporaryFile(max_size=SPOOL_MEMORY_BYTES, mode="w+b") as spool:
        for chunk in _chunks(fetch_chunks, url):
            zip_bytes += len(chunk)
            if zip_bytes > max_zip_bytes:
                raise ArchiveIntegrityError("ZIP exceeds compressed byte limit")
            digest.update(chunk)
            spool.write(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise ArchiveIntegrityError("ZIP SHA-256 does not match official CHECKSUM")
        _check_zip_directory(spool, zip_bytes)
        spool.seek(0)
        try:
            archive = ZipFile(spool)
        except (BadZipFile, LargeZipFile, RuntimeError, EOFError) as exc:
            raise ArchiveIntegrityError("invalid ZIP archive") from exc
        with archive:
            try:
                members = archive.infolist()
                expected_csv = basename[:-4] + ".csv"
                if len(members) != 1 or members[0].filename != expected_csv:
                    raise ArchiveIntegrityError("ZIP must contain exactly its named CSV")
                member = members[0]
                if member.is_dir() or member.flag_bits & 1 or member.file_size > MAX_CSV_BYTES:
                    raise ArchiveIntegrityError("unsafe or oversized ZIP member")
                with archive.open(member) as reader:
                    rows, csv_bytes = _audit_csv(reader, request)
                if csv_bytes != member.file_size:
                    raise ArchiveIntegrityError("ZIP member size does not match parsed CSV")
            except (BadZipFile, LargeZipFile, RuntimeError, EOFError) as exc:
                raise ArchiveIntegrityError("invalid ZIP archive") from exc

            start_s, end_s, expected_rows = _month_limits(request)
            manifest = {
                "source": url,
                "market": request.market,
                "symbol": request.symbol,
                "month": request.month,
                "interval": request.interval,
                "sha256": actual_hash,
                "zip_bytes": zip_bytes,
                "csv_bytes": csv_bytes,
                "rows": rows,
                "expected_rows": expected_rows,
                "first_open_utc": datetime.fromtimestamp(start_s, timezone.utc).isoformat(),
                "last_open_utc": datetime.fromtimestamp(
                    end_s - _INTERVAL_SECONDS[request.interval], timezone.utc,
                ).isoformat(),
                "timestamp_unit": (
                    "microseconds" if request.market == "spot" and request.month >= "2025-01"
                    else "milliseconds"
                ),
                "publication_time_verified": False,
            }
            yield archive, member, manifest


def audit_monthly_klines(
    request: ArchiveRequest,
    *,
    fetch_chunks: ChunkFetcher,
    max_zip_bytes: int = MAX_ZIP_BYTES,
) -> dict[str, object]:
    """Verify one entire month; return metadata, never a price or signal."""
    with _opened_verified_archive(
        request, fetch_chunks=fetch_chunks, max_zip_bytes=max_zip_bytes,
    ) as (_, __, manifest):
        return manifest


@contextmanager
def verified_monthly_klines(
    request: ArchiveRequest,
    *,
    fetch_chunks: ChunkFetcher,
    max_zip_bytes: int = MAX_ZIP_BYTES,
) -> Iterator[Iterator[VerifiedKline]]:
    """Expose bars only after an entire official month passes integrity checks.

    Consume the iterator inside the ``with`` block. It rereads the validated
    temporary ZIP member without redownloading or persisting a price table.
    Each bar remains retrospective research input: archive publication time
    and historical point-in-time visibility are **not** verified.
    """
    with _opened_verified_archive(
        request, fetch_chunks=fetch_chunks, max_zip_bytes=max_zip_bytes,
    ) as (archive, member, manifest):
        unit = 1_000_000 if manifest["timestamp_unit"] == "microseconds" else 1_000
        step = _INTERVAL_SECONDS[request.interval]

        def rows() -> Iterator[VerifiedKline]:
            with archive.open(member) as reader:
                for raw_line in reader:
                    fields = next(csv.reader([raw_line.decode("utf-8-sig")], strict=True))
                    if tuple(fields) == _HEADER:
                        continue
                    opened = datetime.fromtimestamp(int(fields[0]) // unit, timezone.utc)
                    yield VerifiedKline(
                        request=request,
                        open_time=opened,
                        window_end=opened + timedelta(seconds=step),
                        quote_volume=Decimal(fields[7]),
                        taker_buy_quote_volume=Decimal(fields[10]),
                        trade_count=int(fields[8]),
                        archive_sha256=str(manifest["sha256"]),
                        source_url=str(manifest["source"]),
                    )

        iterator = rows()
        try:
            yield iterator
        finally:
            iterator.close()


def audit_months(
    requests: Iterable[ArchiveRequest], *, fetch_chunks: ChunkFetcher,
) -> Iterator[dict[str, object]]:
    """Yield manifests one month at a time; never retain all archive contents."""
    seen: set[ArchiveRequest] = set()
    for request in requests:
        if not isinstance(request, ArchiveRequest):
            raise TypeError("all requests must be ArchiveRequest instances")
        if request in seen:
            raise ValueError("duplicate monthly archive request")
        seen.add(request)
        yield audit_monthly_klines(request, fetch_chunks=fetch_chunks)
