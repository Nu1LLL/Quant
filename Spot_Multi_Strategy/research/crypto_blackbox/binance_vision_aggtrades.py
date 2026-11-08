"""Checksum-gated, one-month-at-a-time Binance Spot aggTrades reader.

This is source plumbing, not a trading rule. Historical publication timing and
executable quotes are unproved. The caller explicitly supplies the transport;
raw ZIP bytes occupy only a bounded temporary spool deleted on exit.
"""

from __future__ import annotations

from calendar import monthrange
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import re
from tempfile import SpooledTemporaryFile
from typing import Iterator
from zipfile import BadZipFile, LargeZipFile, ZipFile

from .binance_vision_5m import (
    ArchiveIntegrityError, ChunkFetcher, _check_zip_directory, _checksum,
    _chunks, http_chunk_fetcher,
)


DEFAULT_MAX_ZIP_BYTES = 512 * 1024 * 1024
HARD_MAX_ZIP_BYTES = 3 * 1024 * 1024 * 1024
MAX_CSV_BYTES = 24 * 1024 * 1024 * 1024
MAX_CSV_LINE_BYTES = 1024
SPOOL_MEMORY_BYTES = 4 * 1024 * 1024
_SYMBOL = re.compile(r"[A-Z0-9]{3,32}\Z")
_MONTH = re.compile(r"(\d{4})-(0[1-9]|1[0-2])\Z")


@dataclass(frozen=True)
class AggTradeRequest:
    symbol: str
    month: str

    def __post_init__(self) -> None:
        if not _SYMBOL.fullmatch(self.symbol):
            raise ValueError("symbol must be uppercase Binance notation")
        if not _MONTH.fullmatch(self.month) or self.month < "2020-01":
            raise ValueError("month must be YYYY-MM from 2020 onward")


@dataclass(frozen=True)
class VerifiedAggTrade:
    """Archive-verified execution event, not an L1 quote or individual fill."""

    symbol: str
    agg_id: int
    first_trade_id: int
    last_trade_id: int
    trade_time_us: int
    price: Decimal
    quantity: Decimal
    buyer_is_maker: bool
    best_price_match: bool
    archive_sha256: str
    archive_publication_time_verified: bool = False


def archive_url(request: AggTradeRequest) -> tuple[str, str]:
    if not isinstance(request, AggTradeRequest):
        raise TypeError("request must be AggTradeRequest")
    basename = f"{request.symbol}-aggTrades-{request.month}.zip"
    return (f"https://data.binance.vision/data/spot/monthly/aggTrades/"
            f"{request.symbol}/{basename}", basename)


def _limits(request: AggTradeRequest) -> tuple[int, int, int, int]:
    year, month = map(int, request.month.split("-"))
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
    end = start + monthrange(year, month)[1] * 86400
    unit = 1_000_000 if request.month >= "2025-01" else 1_000
    digits = 16 if unit == 1_000_000 else 13
    return start * unit, end * unit, unit, digits


def _positive_decimal(value: str, label: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ArchiveIntegrityError(f"{label} is not numeric") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ArchiveIntegrityError(f"{label} is not positive and finite")
    return parsed


def _nonnegative_id(value: str, label: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise ArchiveIntegrityError(f"{label} is not a nonnegative integer")
    return int(value)


def _row(fields: list[str], request: AggTradeRequest, digest: str,
         limits: tuple[int, int, int, int]) -> VerifiedAggTrade:
    if len(fields) != 8:
        raise ArchiveIntegrityError("aggTrades CSV must have eight columns")
    agg_id = _nonnegative_id(fields[0], "aggTrade ID")
    first = _nonnegative_id(fields[3], "first trade ID")
    last = _nonnegative_id(fields[4], "last trade ID")
    if first > last:
        raise ArchiveIntegrityError("aggregate raw-trade ID range is inverted")
    start, end, unit, digits = limits
    stamp_text = fields[5]
    if len(stamp_text) != digits:
        raise ArchiveIntegrityError("incorrect aggTrade timestamp unit")
    stamp = _nonnegative_id(stamp_text, "aggTrade timestamp")
    if not start <= stamp < end:
        raise ArchiveIntegrityError("aggTrade timestamp is outside requested month")
    if fields[6] not in {"True", "False"} or fields[7] not in {"True", "False"}:
        raise ArchiveIntegrityError("maker/best-match flags must be booleans")
    return VerifiedAggTrade(
        request.symbol, agg_id, first, last, stamp * (1_000_000 // unit),
        _positive_decimal(fields[1], "price"),
        _positive_decimal(fields[2], "quantity"),
        fields[6] == "True", fields[7] == "True", digest,
    )


def _read_rows(reader, request: AggTradeRequest, digest: str) -> Iterator[VerifiedAggTrade]:
    previous: VerifiedAggTrade | None = None
    limits = _limits(request)
    while True:
        line = reader.readline(MAX_CSV_LINE_BYTES + 1)
        if not line:
            break
        if len(line) > MAX_CSV_LINE_BYTES:
            raise ArchiveIntegrityError("aggTrades CSV line exceeds byte limit")
        try:
            fields = next(csv.reader([line.decode("utf-8-sig")], strict=True))
        except (UnicodeError, csv.Error, StopIteration) as exc:
            raise ArchiveIntegrityError("malformed aggTrades CSV line") from exc
        row = _row(fields, request, digest, limits)
        if previous is not None:
            if row.agg_id != previous.agg_id + 1:
                raise ArchiveIntegrityError("missing, repeated or reordered aggTrade ID")
            if row.first_trade_id <= previous.last_trade_id:
                raise ArchiveIntegrityError("overlapping or reordered raw-trade IDs")
            if row.trade_time_us < previous.trade_time_us:
                raise ArchiveIntegrityError("aggTrade timestamps moved backward")
        previous = row
        yield row


@contextmanager
def _opened_verified_archive(
    request: AggTradeRequest, *, fetch_chunks: ChunkFetcher,
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
) -> Iterator[tuple[ZipFile, object, dict[str, object]]]:
    if not isinstance(request, AggTradeRequest):
        raise TypeError("request must be AggTradeRequest")
    if type(max_zip_bytes) is not int or not 1 <= max_zip_bytes <= HARD_MAX_ZIP_BYTES:
        raise ValueError("max_zip_bytes is outside the bounded one-month limit")
    url, basename = archive_url(request)
    expected_hash = _checksum(fetch_chunks, url, basename)
    digest = sha256()
    zip_bytes = 0
    with SpooledTemporaryFile(max_size=SPOOL_MEMORY_BYTES, mode="w+b") as spool:
        for chunk in _chunks(fetch_chunks, url):
            zip_bytes += len(chunk)
            if zip_bytes > max_zip_bytes:
                raise ArchiveIntegrityError("monthly ZIP exceeds temporary-storage limit")
            digest.update(chunk)
            spool.write(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise ArchiveIntegrityError("monthly ZIP failed official SHA-256 CHECKSUM")
        _check_zip_directory(spool, zip_bytes)
        spool.seek(0)
        try:
            with ZipFile(spool) as archive:
                members = archive.infolist()
                member = members[0]
                if (len(members) != 1 or member.filename != basename[:-4] + ".csv"
                        or member.is_dir() or member.flag_bits & 1
                        or member.file_size > MAX_CSV_BYTES):
                    raise ArchiveIntegrityError("unsafe or unexpected aggTrades ZIP member")
                count = csv_bytes = 0
                first_row = last_row = None
                with archive.open(member) as reader:
                    # Count uncompressed bytes independently of the ZIP header.
                    class CountingReader:
                        def readline(self, limit: int) -> bytes:
                            nonlocal csv_bytes
                            line = reader.readline(limit)
                            csv_bytes += len(line)
                            if csv_bytes > MAX_CSV_BYTES:
                                raise ArchiveIntegrityError("aggTrades CSV exceeds byte limit")
                            return line

                    for row in _read_rows(CountingReader(), request, actual_hash):
                        count += 1
                        first_row = row if first_row is None else first_row
                        last_row = row
                if count == 0 or csv_bytes != member.file_size:
                    raise ArchiveIntegrityError("empty or truncated aggTrades CSV member")
                manifest = {
                    "source": url, "symbol": request.symbol, "month": request.month,
                    "sha256": actual_hash, "zip_bytes": zip_bytes,
                    "csv_bytes": csv_bytes, "rows": count,
                    "first_agg_id": first_row.agg_id,
                    "last_agg_id": last_row.agg_id,
                    "first_raw_trade_id": first_row.first_trade_id,
                    "last_raw_trade_id": last_row.last_trade_id,
                    "first_trade_time_us": first_row.trade_time_us,
                    "last_trade_time_us": last_row.trade_time_us,
                    "timestamp_unit": (
                        "microseconds" if request.month >= "2025-01" else "milliseconds"),
                    "publication_time_verified": False,
                }
                yield archive, member, manifest
        except (BadZipFile, LargeZipFile, RuntimeError, EOFError) as exc:
            raise ArchiveIntegrityError("invalid aggTrades ZIP archive") from exc


def audit_month(
    request: AggTradeRequest, *, fetch_chunks: ChunkFetcher,
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
) -> dict[str, object]:
    """Return source integrity metadata only, never trade values or returns."""
    with _opened_verified_archive(
        request, fetch_chunks=fetch_chunks, max_zip_bytes=max_zip_bytes,
    ) as (_, __, manifest):
        return manifest


@contextmanager
def verified_month(
    request: AggTradeRequest, *, fetch_chunks: ChunkFetcher,
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
) -> Iterator[Iterator[VerifiedAggTrade]]:
    """Expose rows only after the entire month has passed source validation."""
    with _opened_verified_archive(
        request, fetch_chunks=fetch_chunks, max_zip_bytes=max_zip_bytes,
    ) as (archive, member, manifest):
        def rows() -> Iterator[VerifiedAggTrade]:
            with archive.open(member) as reader:
                yield from _read_rows(reader, request, str(manifest["sha256"]))

        iterator = rows()
        try:
            yield iterator
        finally:
            iterator.close()


def verify_adjacent_months(left: dict[str, object], right: dict[str, object]) -> None:
    """Reject apparent cross-month ID/time gaps in same-symbol manifests."""
    if left["symbol"] != right["symbol"]:
        raise ArchiveIntegrityError("adjacent manifests use different symbols")
    year, month = map(int, str(left["month"]).split("-"))
    next_month = f"{year + (month == 12)}-{(month % 12) + 1:02d}"
    if right["month"] != next_month:
        raise ArchiveIntegrityError("source months are not adjacent")
    if int(right["first_agg_id"]) != int(left["last_agg_id"]) + 1:
        raise ArchiveIntegrityError("aggTrade ID gap across months")
    if int(right["first_raw_trade_id"]) <= int(left["last_raw_trade_id"]):
        raise ArchiveIntegrityError("raw-trade IDs overlap across months")
    if int(right["first_trade_time_us"]) < int(left["last_trade_time_us"]):
        raise ArchiveIntegrityError("trade timestamps move backward across months")
