"""Pinned, bounded Binance Vision feed for the BTC/ETH pairs experiment.

Both monthly kline ZIPs are fully verified against their official CHECKSUMs,
audited for their complete minute grid and OHLC, and checked against the
pre-result manifest before that month's first price is exposed. Funding ZIPs
receive the same dual hash check; their actual millisecond timestamps are
retained rather than rounded to the eight-hour settlement anchors.

This module is a research data reader. It writes no archive and calculates no
signals, positions or returns. Historical archive publication time and
executable bid/ask prices are not established by these checks.
"""

from __future__ import annotations

from calendar import monthrange
import csv
from dataclasses import dataclass
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import io
from itertools import zip_longest
import math
from pathlib import Path
import re
from typing import Iterator, MutableMapping
from zipfile import BadZipFile, LargeZipFile, ZipFile

from .crypto_blackbox.binance_vision_5m import (
    ArchiveIntegrityError,
    ArchiveRequest,
    ChunkFetcher,
    _check_zip_directory,
    _checksum,
    _chunks,
    _opened_verified_archive,
    http_chunk_fetcher,
)


KLINE_MANIFEST_PATH = (
    Path(__file__).parent / "preregistrations"
    / "2026-11-07_btc_eth_pairs_kline_manifest.csv"
)
FUNDING_MANIFEST_PATH = (
    Path(__file__).parent / "preregistrations"
    / "2026-11-07_btc_eth_pairs_funding_manifest.csv"
)
_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_KLINE_FIELDS = (
    "month", "market", "symbol", "interval", "sha256", "rows",
    "missing", "duplicate",
)
_FUNDING_MANIFEST_FIELDS = (
    "month", "market", "symbol", "dataset", "sha256", "rows",
)
_FUNDING_CSV_FIELDS = (
    "calc_time", "funding_interval_hours", "last_funding_rate",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FUNDING_ZIP_LIMIT = 64 * 1024
_FUNDING_CSV_LIMIT = 128 * 1024
_MINUTE_MS = 60_000
_EIGHT_HOURS_MS = 8 * 60 * _MINUTE_MS


class PairDataError(ValueError):
    """A fixed input, checksum, schema or time-grid requirement failed."""


@dataclass(frozen=True, slots=True)
class MinutePairBar:
    """Synchronized UTC minute-open OHLC from two fully audited archives."""

    time: datetime
    btc_open: float
    btc_high: float
    btc_low: float
    btc_close: float
    eth_open: float
    eth_high: float
    eth_low: float
    eth_close: float


@dataclass(frozen=True, slots=True)
class FundingEvent:
    """Actual archived funding time and signed rate for one contract."""

    time: datetime
    symbol: str
    rate: float


def _months(first_year: int, last_year: int) -> tuple[str, ...]:
    return tuple(
        f"{year}-{month:02d}"
        for year in range(first_year, last_year + 1)
        for month in range(1, 13)
    )


def _manifest_rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != fields:
                raise PairDataError("unexpected pinned manifest columns")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PairDataError(f"pinned manifest unavailable or malformed: {path}") from exc
    return rows


def read_kline_manifest(
    path: Path = KLINE_MANIFEST_PATH,
) -> dict[tuple[str, str], dict[str, str]]:
    """Require exactly 72 months and both fixed contracts, in frozen order."""
    rows = _manifest_rows(path, _KLINE_FIELDS)
    expected = tuple((month, symbol) for month in _months(2020, 2025)
                     for symbol in _SYMBOLS)
    if len(rows) != len(expected):
        raise PairDataError("kline manifest must contain exactly 144 pins")
    pins: dict[tuple[str, str], dict[str, str]] = {}
    for row, (month, symbol) in zip(rows, expected):
        year, number = map(int, month.split("-"))
        expected_count = monthrange(year, number)[1] * 1440
        if (
            set(row) != set(_KLINE_FIELDS)
            or any(row[field] is None or row[field] == "" for field in _KLINE_FIELDS)
            or row["month"] != month
            or row["market"] != "um"
            or row["symbol"] != symbol
            or row["interval"] != "1m"
            or not _SHA.fullmatch(row["sha256"])
            or row["rows"] != str(expected_count)
            or row["missing"] != "0"
            or row["duplicate"] != "0"
        ):
            raise PairDataError(f"invalid kline manifest pin: {month} {symbol}")
        pins[(month, symbol)] = row
    return pins


def read_funding_manifest(
    path: Path = FUNDING_MANIFEST_PATH,
) -> dict[tuple[str, str], dict[str, str]]:
    """Require every fixed 8-hour funding month and both contract hashes."""
    rows = _manifest_rows(path, _FUNDING_MANIFEST_FIELDS)
    expected = tuple((month, symbol) for month in _months(2021, 2025)
                     for symbol in _SYMBOLS)
    if len(rows) != len(expected):
        raise PairDataError("funding manifest must contain exactly 120 pins")
    pins: dict[tuple[str, str], dict[str, str]] = {}
    for row, (month, symbol) in zip(rows, expected):
        year, number = map(int, month.split("-"))
        expected_count = monthrange(year, number)[1] * 3
        if (
            set(row) != set(_FUNDING_MANIFEST_FIELDS)
            or any(row[field] is None or row[field] == ""
                   for field in _FUNDING_MANIFEST_FIELDS)
            or row["month"] != month
            or row["market"] != "um"
            or row["symbol"] != symbol
            or row["dataset"] != "fundingRate"
            or not _SHA.fullmatch(row["sha256"])
            or row["rows"] != str(expected_count)
        ):
            raise PairDataError(f"invalid funding manifest pin: {month} {symbol}")
        pins[(month, symbol)] = row
    return pins


def _price_row(fields: list[str], expected_time_ms: int) -> tuple[float, ...]:
    """Check a second read of an already audited member before yielding it."""
    if len(fields) != 12 or fields[0] != str(expected_time_ms):
        raise PairDataError("audited kline changed or lost its minute alignment")
    try:
        opening, high, low, closing = (float(fields[index]) for index in (1, 2, 3, 4))
    except (ValueError, OverflowError) as exc:
        raise PairDataError("invalid audited kline price") from exc
    if (
        not all(math.isfinite(value) and value > 0
                for value in (opening, high, low, closing))
        or high < max(opening, closing, low)
        or low > min(opening, closing, high)
    ):
        raise PairDataError("invalid audited kline OHLC on second read")
    return opening, high, low, closing


def _bar_rows(archive: ZipFile, member: object) -> Iterator[list[str]]:
    with archive.open(member) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            for fields in reader:
                if fields and fields[0] == "open_time":
                    continue
                yield fields


def iter_paired_bars(
    *,
    fetch_chunks: ChunkFetcher = http_chunk_fetcher,
    manifest_path: Path = KLINE_MANIFEST_PATH,
    audit_hashes: MutableMapping[tuple[str, str], str] | None = None,
) -> Iterator[MinutePairBar]:
    """Stream all 2020–25 paired 1m bars; hold only two monthly ZIP spools.

    Each symbol's whole month passes the existing strict CSV/ZIP audit, the
    current official CHECKSUM and the committed manifest before any bar from
    that month is yielded. Close the returned generator if consumption stops
    early, so both temporary spools are released immediately.
    """
    pins = read_kline_manifest(manifest_path)
    for month in _months(2020, 2025):
        btc = ArchiveRequest("um", "BTCUSDT", month, "1m")
        eth = ArchiveRequest("um", "ETHUSDT", month, "1m")
        try:
            with _opened_verified_archive(
                btc, fetch_chunks=fetch_chunks,
            ) as (btc_zip, btc_member, btc_audit), _opened_verified_archive(
                eth, fetch_chunks=fetch_chunks,
            ) as (eth_zip, eth_member, eth_audit):
                for symbol, audit in (("BTCUSDT", btc_audit), ("ETHUSDT", eth_audit)):
                    pin = pins[(month, symbol)]
                    if audit["sha256"] != pin["sha256"] or audit["rows"] != int(pin["rows"]):
                        raise PairDataError(f"kline ZIP differs from pin: {month} {symbol}")
                if audit_hashes is not None:
                    audit_hashes[(month, "BTCUSDT")] = str(btc_audit["sha256"])
                    audit_hashes[(month, "ETHUSDT")] = str(eth_audit["sha256"])
                year, number = map(int, month.split("-"))
                expected_ms = int(datetime(year, number, 1, tzinfo=timezone.utc)
                                  .timestamp() * 1000)
                expected_rows = monthrange(year, number)[1] * 1440
                count = 0
                with closing(_bar_rows(btc_zip, btc_member)) as btc_rows, closing(
                    _bar_rows(eth_zip, eth_member)
                ) as eth_rows:
                    for btc_fields, eth_fields in zip_longest(btc_rows, eth_rows):
                        if btc_fields is None or eth_fields is None or count >= expected_rows:
                            raise PairDataError(f"paired minute grid differs after audit: {month}")
                        btc_prices = _price_row(btc_fields, expected_ms)
                        eth_prices = _price_row(eth_fields, expected_ms)
                        yield MinutePairBar(
                            datetime.fromtimestamp(expected_ms // 1000, timezone.utc),
                            *btc_prices, *eth_prices,
                        )
                        count += 1
                        expected_ms += _MINUTE_MS
                if count != expected_rows:
                    raise PairDataError(f"paired minute month incomplete: {month}")
        except (ArchiveIntegrityError, BadZipFile, LargeZipFile,
                RuntimeError, EOFError, UnicodeError, csv.Error, OSError) as exc:
            raise PairDataError(f"invalid paired kline archive: {month}") from exc


def _funding_url(month: str, symbol: str) -> tuple[str, str]:
    if symbol not in _SYMBOLS:
        raise ValueError("funding symbol is outside the fixed pair")
    basename = f"{symbol}-fundingRate-{month}.zip"
    return (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
        f"{symbol}/{basename}",
        basename,
    )


def _actual_utc_datetime(stamp_ms: int) -> datetime:
    return (datetime.fromtimestamp(stamp_ms // 1000, timezone.utc)
            + timedelta(milliseconds=stamp_ms % 1000))


def _read_funding_month(
    month: str,
    symbol: str,
    pin: dict[str, str],
    fetch_chunks: ChunkFetcher,
) -> tuple[list[FundingEvent], str]:
    """Read one small pinned funding ZIP and demand every exact UTC anchor."""
    url, basename = _funding_url(month, symbol)
    try:
        official_hash = _checksum(fetch_chunks, url, basename)
        digest = sha256()
        payload = bytearray()
        for chunk in _chunks(fetch_chunks, url):
            if len(payload) + len(chunk) > _FUNDING_ZIP_LIMIT:
                raise PairDataError("funding ZIP exceeds compressed byte limit")
            digest.update(chunk)
            payload.extend(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != official_hash or actual_hash != pin["sha256"]:
            raise PairDataError(f"funding ZIP differs from official CHECKSUM or pin: {month} {symbol}")
        _check_zip_directory(io.BytesIO(payload), len(payload))
        with ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if (
                len(members) != 1
                or members[0].filename != basename[:-4] + ".csv"
                or members[0].is_dir()
                or members[0].flag_bits & 1
                or members[0].file_size > _FUNDING_CSV_LIMIT
            ):
                raise PairDataError("funding ZIP has an unexpected member")
            year, number = map(int, month.split("-"))
            start_ms = int(datetime(year, number, 1, tzinfo=timezone.utc)
                           .timestamp() * 1000)
            expected_count = monthrange(year, number)[1] * 3
            events: list[FundingEvent] = []
            with archive.open(members[0]) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle, strict=True)
                    if tuple(next(reader, ())) != _FUNDING_CSV_FIELDS:
                        raise PairDataError("unexpected funding CSV schema")
                    for index, fields in enumerate(reader):
                        if index >= expected_count or len(fields) != 3:
                            raise PairDataError("extra or malformed funding event")
                        if len(fields[0]) != 13 or not fields[0].isascii() or not fields[0].isdigit():
                            raise PairDataError("invalid funding millisecond timestamp")
                        try:
                            stamp_ms = int(fields[0])
                            interval = Decimal(fields[1])
                            decimal_rate = Decimal(fields[2])
                            rate = float(decimal_rate)
                        except (ValueError, InvalidOperation, OverflowError) as exc:
                            raise PairDataError("invalid funding numeric field") from exc
                        anchor_ms = start_ms + index * _EIGHT_HOURS_MS
                        if not anchor_ms <= stamp_ms < anchor_ms + 1000:
                            raise PairDataError(
                                f"funding event missing, duplicated or off anchor: {month} {symbol} #{index}"
                            )
                        if (not interval.is_finite() or interval <= 0
                                or not decimal_rate.is_finite() or not math.isfinite(rate)):
                            raise PairDataError("invalid funding interval or nonfinite rate")
                        events.append(FundingEvent(
                            _actual_utc_datetime(stamp_ms), symbol, rate,
                        ))
            if len(events) != expected_count or len(events) != int(pin["rows"]):
                raise PairDataError(f"funding month incomplete: {month} {symbol}")
            return events, actual_hash
    except (ArchiveIntegrityError, BadZipFile, LargeZipFile,
            RuntimeError, EOFError, UnicodeError, csv.Error, OSError) as exc:
        raise PairDataError(f"invalid funding archive: {month} {symbol}") from exc


def load_funding_events(
    *,
    fetch_chunks: ChunkFetcher = http_chunk_fetcher,
    manifest_path: Path = FUNDING_MANIFEST_PATH,
    audit_hashes: MutableMapping[tuple[str, str], str] | None = None,
) -> tuple[FundingEvent, ...]:
    """Load both contracts' 2021–25 funding, preserving actual millisecond times.

    The month-by-month exact 8-hour anchor check excludes missing, duplicate,
    delayed-by-seconds, unsorted, and extra funding events. The returned list
    is small (10,956 records) and sorted by actual time and symbol.
    """
    pins = read_funding_manifest(manifest_path)
    result: list[FundingEvent] = []
    for month in _months(2021, 2025):
        for symbol in _SYMBOLS:
            events, digest = _read_funding_month(
                month, symbol, pins[(month, symbol)], fetch_chunks,
            )
            result.extend(events)
            if audit_hashes is not None:
                audit_hashes[(month, symbol)] = digest
    result.sort(key=lambda item: (item.time, item.symbol))
    return tuple(result)
