"""Pinned, read-only three-altcoin minute/flow/funding archive feed.

Only the separately committed 690a7f5 manifests are accepted. Monthly ZIPs
are verified against the current official CHECKSUM and the frozen SHA before
their first row is exposed. No raw archive is retained after its month.
"""

from __future__ import annotations

from calendar import monthrange
from contextlib import ExitStack
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import io
from itertools import zip_longest
import math
from pathlib import Path
import re
from typing import Iterator, MutableMapping
from zipfile import BadZipFile, ZipFile

from .crypto_blackbox.binance_vision_5m import (
    ArchiveIntegrityError, ArchiveRequest, ChunkFetcher, _check_zip_directory,
    _checksum, _chunks, _opened_verified_archive, http_chunk_fetcher,
)

SYMBOLS = ("XRPUSDT", "ADAUSDT", "DOGEUSDT")
KLINE_MANIFEST = Path(__file__).parent / "preregistrations/2026-11-07_xrp_ada_doge_flow_kline_manifest.csv"
FUNDING_MANIFEST = Path(__file__).parent / "preregistrations/2026-11-07_xrp_ada_doge_flow_funding_manifest.csv"
_FIELDS = ("month", "symbol", "market", "dataset", "sha256")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MINUTE_MS = 60_000
_HOURS_8_MS = 8 * 60 * _MINUTE_MS


class AltFlowDataError(ValueError):
    """A frozen data-integrity requirement failed; discard partial outputs."""


@dataclass(frozen=True, slots=True)
class AltMinute:
    time: datetime
    # Each tuple is in the fixed SYMBOLS order, not selected after results.
    opening: tuple[float, float, float]
    high: tuple[float, float, float]
    low: tuple[float, float, float]
    close: tuple[float, float, float]
    quote: tuple[float, float, float]
    taker_buy_quote: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class AltFunding:
    time: datetime
    symbol: str
    rate: float


def _months(start: tuple[int, int], end: tuple[int, int]) -> tuple[str, ...]:
    return tuple(f"{y}-{m:02d}" for y in range(start[0], end[0] + 1)
                 for m in range(1, 13) if start <= (y, m) <= end)


def read_manifest(path: Path, *, kind: str) -> dict[tuple[str, str], str]:
    if kind not in {"kline", "funding"}:
        raise ValueError("unknown manifest kind")
    months = _months((2020, 9), (2025, 12)) if kind == "kline" else _months((2021, 1), (2025, 12))
    expected = [(m, s) for m in months for s in SYMBOLS]
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, strict=True)
            if tuple(reader.fieldnames or ()) != _FIELDS:
                raise AltFlowDataError("manifest columns changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AltFlowDataError("manifest unavailable or malformed") from exc
    if len(rows) != len(expected):
        raise AltFlowDataError("manifest is missing fixed symbol-months")
    pins = {}
    dataset = "1m" if kind == "kline" else "fundingRate"
    for row, (month, symbol) in zip(rows, expected):
        if (set(row) != set(_FIELDS) or row["month"] != month or row["symbol"] != symbol
                or row["market"] != "um" or row["dataset"] != dataset
                or not _SHA.fullmatch(row["sha256"] or "")):
            raise AltFlowDataError(f"manifest pin changed: {month} {symbol}")
        pins[(month, symbol)] = row["sha256"]
    return pins


def _row_iter(archive: ZipFile, member: object) -> Iterator[list[str]]:
    with archive.open(member) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            for fields in reader:
                if fields and fields[0] == "open_time":
                    continue
                yield fields


def _parse_price(fields: list[str], expected_ms: int) -> tuple[float, ...]:
    if len(fields) != 12 or fields[0] != str(expected_ms):
        raise AltFlowDataError("minute grid changed after archive audit")
    try:
        o, h, l, c, q, b = (float(fields[j]) for j in (1, 2, 3, 4, 7, 10))
        close_ms, trade_count = int(fields[6]), int(fields[8])
    except (ValueError, OverflowError) as exc:
        raise AltFlowDataError("invalid minute price or quote amount") from exc
    if (not all(math.isfinite(x) for x in (o, h, l, c, q, b))
            or min(o, h, l, c) <= 0 or h < max(o, c, l)
            or l > min(o, c, h) or not 0 <= b <= q
            or close_ms != expected_ms + _MINUTE_MS - 1 or trade_count < 0):
        raise AltFlowDataError("invalid OHLC or taker-quote volume")
    return o, h, l, c, q, b


def iter_minutes(*, fetch_chunks: ChunkFetcher = http_chunk_fetcher,
                 audit_hashes: MutableMapping[tuple[str, str], str] | None = None
                 ) -> Iterator[AltMinute]:
    """Yield synchronized fixed-universe minutes from all 64 full months."""
    pins = read_manifest(KLINE_MANIFEST, kind="kline")
    for month in _months((2020, 9), (2025, 12)):
        try:
            with ExitStack() as stack:
                opened = [stack.enter_context(_opened_verified_archive(
                    ArchiveRequest("um", symbol, month, "1m"), fetch_chunks=fetch_chunks,
                )) for symbol in SYMBOLS]
                for symbol, (_, _, audit) in zip(SYMBOLS, opened):
                    if audit["sha256"] != pins[(month, symbol)]:
                        raise AltFlowDataError(f"K-line hash differs from preregistration: {month} {symbol}")
                    if audit_hashes is not None:
                        audit_hashes[(month, symbol)] = str(audit["sha256"])
                y, m = map(int, month.split("-"))
                expected_ms = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)
                expected_count = monthrange(y, m)[1] * 1440
                count = 0
                for rows in zip_longest(*(_row_iter(z, member) for z, member, _ in opened)):
                    if None in rows or count >= expected_count:
                        raise AltFlowDataError(f"three-coin minute grid changed: {month}")
                    parsed = tuple(_parse_price(row, expected_ms) for row in rows)
                    yield AltMinute(
                        datetime.fromtimestamp(expected_ms // 1000, timezone.utc),
                        tuple(p[0] for p in parsed), tuple(p[1] for p in parsed),
                        tuple(p[2] for p in parsed), tuple(p[3] for p in parsed),
                        tuple(p[4] for p in parsed), tuple(p[5] for p in parsed),
                    )
                    count += 1
                    expected_ms += _MINUTE_MS
                if count != expected_count:
                    raise AltFlowDataError(f"incomplete three-coin month: {month}")
        except (ArchiveIntegrityError, BadZipFile, RuntimeError, EOFError,
                UnicodeError, csv.Error, OSError) as exc:
            raise AltFlowDataError(f"invalid three-coin archive: {month}") from exc


def _funding_month(month: str, symbol: str, pin: str, fetch_chunks: ChunkFetcher
                   ) -> tuple[list[AltFunding], str]:
    name = f"{symbol}-fundingRate-{month}.zip"
    url = ("https://data.binance.vision/data/futures/um/monthly/fundingRate/"
           f"{symbol}/{name}")
    official = _checksum(fetch_chunks, url, name)
    payload = bytearray()
    for chunk in _chunks(fetch_chunks, url):
        if len(payload) + len(chunk) > 64 * 1024:
            raise AltFlowDataError("funding ZIP exceeds 64 KiB")
        payload.extend(chunk)
    digest = sha256(payload).hexdigest()
    if digest != official or digest != pin:
        raise AltFlowDataError(f"funding hash mismatch: {month} {symbol}")
    try:
        _check_zip_directory(io.BytesIO(payload), len(payload))
        with ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].file_size > 128 * 1024 or members[0].flag_bits & 1):
                raise AltFlowDataError("invalid funding ZIP member")
            y, m = map(int, month.split("-"))
            first_ms = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)
            expected_count = monthrange(y, m)[1] * 3
            with archive.open(members[0]) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                if tuple(next(reader, ())) != ("calc_time", "funding_interval_hours", "last_funding_rate"):
                    raise AltFlowDataError("funding CSV schema changed")
                result = []
                for index, fields in enumerate(reader):
                    if index >= expected_count or len(fields) != 3:
                        raise AltFlowDataError("extra or malformed funding event")
                    try:
                        ms = int(fields[0]); interval = Decimal(fields[1]); decimal_rate = Decimal(fields[2])
                        rate = float(decimal_rate)
                    except (ValueError, InvalidOperation, OverflowError) as exc:
                        raise AltFlowDataError("funding numeric field invalid") from exc
                    anchor = first_ms + index * _HOURS_8_MS
                    if (not anchor <= ms < anchor + 1000 or not interval.is_finite()
                            or interval <= 0 or not decimal_rate.is_finite()
                            or not math.isfinite(rate)):
                        raise AltFlowDataError("funding anchor or rate invalid")
                    time = datetime.fromtimestamp(ms // 1000, timezone.utc) + timedelta(milliseconds=ms % 1000)
                    result.append(AltFunding(time, symbol, rate))
                if len(result) != expected_count:
                    raise AltFlowDataError("funding month incomplete")
            return result, digest
    except (BadZipFile, EOFError, RuntimeError, UnicodeError, csv.Error) as exc:
        raise AltFlowDataError("funding ZIP/CSV invalid") from exc


def load_funding(*, fetch_chunks: ChunkFetcher = http_chunk_fetcher,
                 audit_hashes: MutableMapping[tuple[str, str], str] | None = None
                 ) -> tuple[AltFunding, ...]:
    """Load only fixed OOS archived rates; preserve exact millisecond times."""
    pins = read_manifest(FUNDING_MANIFEST, kind="funding")
    all_events: list[AltFunding] = []
    for month in _months((2021, 1), (2025, 12)):
        for symbol in SYMBOLS:
            events, digest = _funding_month(month, symbol, pins[(month, symbol)], fetch_chunks)
            all_events.extend(events)
            if audit_hashes is not None:
                audit_hashes[(month, symbol)] = digest
    all_events.sort(key=lambda x: (x.time, x.symbol))
    return tuple(all_events)
