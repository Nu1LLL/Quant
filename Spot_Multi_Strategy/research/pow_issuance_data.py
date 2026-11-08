"""Pinned three-PoW-token hourly and funding archive reader.

Whole monthly archives are validated against both the committed digest and
fresh official CHECKSUM before any of their price or rate rows are yielded.
Only the terminal first open and first funding event from January 2026 are exposed.
"""

from __future__ import annotations

from calendar import monthrange
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import io
import math
from pathlib import Path
import re
from typing import Iterator, MutableMapping
from zipfile import BadZipFile, ZipFile

from .basis_hourly_preflight import _fetch


UTC = timezone.utc
SYMBOLS = ("BCHUSDT", "DOGEUSDT", "ETCUSDT")
KLINE_MANIFEST = Path(__file__).parent / "preregistrations/2026-11-08_three_pow_issuance_1h_manifest.csv"
FUNDING_MANIFEST = Path(__file__).parent / "preregistrations/2026-11-08_three_pow_issuance_fundingRate_manifest.csv"
FIELDS = ("month", "market", "symbol", "dataset", "sha256")


class PowIssuanceDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HourBar:
    time: datetime
    opening: tuple[float, float, float]
    high: tuple[float, float, float]
    low: tuple[float, float, float]
    close: tuple[float, float, float]
    quote_volume: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TerminalOpen:
    time: datetime
    opening: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class FundingEvent:
    time: datetime
    symbol: str
    rate: float


def _months(first: tuple[int, int], last: tuple[int, int]) -> tuple[str, ...]:
    return tuple(f"{y}-{m:02d}" for y in range(first[0], last[0] + 1)
                 for m in range(1, 13) if first <= (y, m) <= last)


def _pins(path: Path, *, kind: str) -> dict[tuple[str, str], str]:
    if kind not in {"1h", "fundingRate"}:
        raise ValueError("unknown archive kind")
    first = (2020, 12) if kind == "1h" else (2021, 1)
    expected = [(m, s) for m in _months(first, (2026, 1)) for s in SYMBOLS]
    column = "hours" if kind == "1h" else "events"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != FIELDS + (column,):
                raise PowIssuanceDataError("pinned manifest schema changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PowIssuanceDataError("pinned manifest unavailable") from exc
    if len(rows) != len(expected):
        raise PowIssuanceDataError("pinned manifest has wrong number of months")
    pins = {}
    for row, (month, symbol) in zip(rows, expected):
        y, m = map(int, month.split("-"))
        expected_count = monthrange(y, m)[1] * (24 if kind == "1h" else 3)
        if (set(row) != set(FIELDS + (column,)) or row["month"] != month
                or row["market"] != "um" or row["symbol"] != symbol
                or row["dataset"] != kind
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"] or "")
                or row[column] != str(expected_count)):
            raise PowIssuanceDataError(f"pinned manifest row changed: {month} {symbol}")
        pins[(month, symbol)] = row["sha256"]
    return pins


def _verified_zip(symbol: str, month: str, kind: str, pin: str) -> ZipFile:
    if kind == "1h":
        name = f"{symbol}-1h-{month}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1h/{name}"
        limit = 2_000_000
    else:
        name = f"{symbol}-fundingRate-{month}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{name}"
        limit = 200_000
    statement = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise PowIssuanceDataError(f"official CHECKSUM malformed: {kind} {symbol} {month}")
    payload = _fetch(url, limit=limit)
    digest = sha256(payload).hexdigest()
    if digest != pin or digest != match.group(1).lower():
        raise PowIssuanceDataError(f"archive digest differs from pinned or official hash: {kind} {symbol} {month}")
    try:
        archive = ZipFile(io.BytesIO(payload))
        members = archive.infolist()
        if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                or members[0].flag_bits & 1
                or members[0].file_size > (5_000_000 if kind == "1h" else 500_000)):
            archive.close()
            raise PowIssuanceDataError(f"unexpected ZIP member: {kind} {symbol} {month}")
        return archive
    except BadZipFile as exc:
        raise PowIssuanceDataError(f"bad ZIP: {kind} {symbol} {month}") from exc


def _price_month(symbol: str, month: str, pin: str) -> tuple[tuple[float, ...], ...]:
    y, m = map(int, month.split("-"))
    expected = int(datetime(y, m, 1, tzinfo=UTC).timestamp() * 1000)
    required = monthrange(y, m)[1] * 24
    rows = []
    first_line = True
    try:
        with _verified_zip(symbol, month, "1h", pin) as archive:
            with archive.open(archive.infolist()[0]) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                for fields in reader:
                    if first_line and fields and fields[0] == "open_time":
                        first_line = False
                        continue
                    first_line = False
                    if (len(fields) != 12 or fields[0] != str(expected)
                            or fields[6] != str(expected + 3_600_000 - 1)):
                        raise PowIssuanceDataError(f"missing or malformed hourly row: {symbol} {month}")
                    try:
                        o, h, l, c = (float(fields[j]) for j in (1, 2, 3, 4))
                        volume, quote, taker_base, taker_quote = (
                            float(fields[j]) for j in (5, 7, 9, 10))
                        count = int(fields[8])
                    except (ValueError, OverflowError) as exc:
                        raise PowIssuanceDataError("invalid hourly numeric field") from exc
                    if (not all(math.isfinite(x) for x in (o, h, l, c, volume, quote,
                                                            taker_base, taker_quote))
                            or min(o, h, l, c) <= 0 or h < max(o, c, l)
                            or l > min(o, c, h) or min(volume, quote, taker_base,
                                                       taker_quote) < 0
                            or taker_base > volume or taker_quote > quote or count < 0):
                        raise PowIssuanceDataError(f"invalid hourly OHLC or volume: {symbol} {month}")
                    rows.append((o, h, l, c, quote))
                    expected += 3_600_000
    except (BadZipFile, UnicodeError, csv.Error, OSError) as exc:
        raise PowIssuanceDataError(f"invalid hourly archive: {symbol} {month}") from exc
    if len(rows) != required:
        raise PowIssuanceDataError(f"incomplete hourly archive: {symbol} {month}")
    return tuple(rows)


def _terminal_month(symbol: str, month: str, pin: str) -> float:
    if month != "2026-01":
        raise ValueError("only the January 2026 terminal open is allowed")
    try:
        with _verified_zip(symbol, month, "1h", pin) as archive:
            with archive.open(archive.infolist()[0]) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""),
                                    strict=True)
                first = next(reader, ())
                if first and first[0] == "open_time":
                    first = next(reader, ())
        stamp = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
        if len(first) != 12 or first[0] != str(stamp) or first[6] != str(stamp + 3_600_000 - 1):
            raise PowIssuanceDataError(f"terminal first hour malformed: {symbol}")
        opening = float(first[1])
        if not math.isfinite(opening) or opening <= 0:
            raise PowIssuanceDataError(f"terminal opening invalid: {symbol}")
        return opening
    except (BadZipFile, UnicodeError, csv.Error, OSError, ValueError) as exc:
        if isinstance(exc, PowIssuanceDataError):
            raise
        raise PowIssuanceDataError(f"invalid terminal archive: {symbol}") from exc


def iter_hours(*, audit_hashes: MutableMapping[tuple[str, str], str] | None = None
               ) -> Iterator[HourBar | TerminalOpen]:
    pins = _pins(KLINE_MANIFEST, kind="1h")
    for month in _months((2020, 12), (2026, 1)):
        if month == "2026-01":
            opens = [_terminal_month(symbol, month, pins[(month, symbol)])
                     for symbol in SYMBOLS]
            if audit_hashes is not None:
                for symbol in SYMBOLS:
                    audit_hashes[(month, symbol)] = pins[(month, symbol)]
            yield TerminalOpen(datetime(2026, 1, 1, tzinfo=UTC), tuple(opens))
            break
        triples = [_price_month(symbol, month, pins[(month, symbol)])
                   for symbol in SYMBOLS]
        if audit_hashes is not None:
            for symbol in SYMBOLS:
                audit_hashes[(month, symbol)] = pins[(month, symbol)]
        y, m = map(int, month.split("-"))
        first = datetime(y, m, 1, tzinfo=UTC)
        for index in range(len(triples[0])):
            yield HourBar(first + timedelta(hours=index),
                          tuple(x[index][0] for x in triples),
                          tuple(x[index][1] for x in triples),
                          tuple(x[index][2] for x in triples),
                          tuple(x[index][3] for x in triples),
                          tuple(x[index][4] for x in triples))


def _funding_month(symbol: str, month: str, pin: str) -> tuple[FundingEvent, ...]:
    y, m = map(int, month.split("-"))
    expected = int(datetime(y, m, 1, tzinfo=UTC).timestamp() * 1000)
    required = monthrange(y, m)[1] * 3
    events = []
    try:
        with _verified_zip(symbol, month, "fundingRate", pin) as archive:
            with archive.open(archive.infolist()[0]) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                if tuple(next(reader, ())) != ("calc_time", "funding_interval_hours", "last_funding_rate"):
                    raise PowIssuanceDataError(f"funding schema changed: {symbol} {month}")
                for fields in reader:
                    if len(fields) != 3:
                        raise PowIssuanceDataError(f"funding field count changed: {symbol} {month}")
                    try:
                        ms = int(fields[0]); interval = Decimal(fields[1]); rate_decimal = Decimal(fields[2])
                        rate = float(rate_decimal)
                    except (ValueError, InvalidOperation, OverflowError) as exc:
                        raise PowIssuanceDataError("invalid funding numeric field") from exc
                    if (not expected <= ms < expected + 1000
                            or not interval.is_finite() or interval != 8
                            or not rate_decimal.is_finite() or not math.isfinite(rate)):
                        raise PowIssuanceDataError(f"funding grid or rate invalid: {symbol} {month}")
                    time = datetime.fromtimestamp(ms // 1000, UTC) + timedelta(milliseconds=ms % 1000)
                    events.append(FundingEvent(time, symbol, rate))
                    expected += 8 * 3_600_000
                    if month == "2026-01":
                        break
    except (BadZipFile, UnicodeError, csv.Error, OSError) as exc:
        raise PowIssuanceDataError(f"invalid funding archive: {symbol} {month}") from exc
    if len(events) != (1 if month == "2026-01" else required):
        raise PowIssuanceDataError(f"incomplete funding archive: {symbol} {month}")
    return tuple(events)


def load_funding(*, audit_hashes: MutableMapping[tuple[str, str], str] | None = None
                 ) -> tuple[FundingEvent, ...]:
    pins = _pins(FUNDING_MANIFEST, kind="fundingRate")
    result = []
    for month in _months((2021, 1), (2026, 1)):
        for symbol in SYMBOLS:
            events = _funding_month(symbol, month, pins[(month, symbol)])
            result.extend(events[:1] if month == "2026-01" else events)
            if audit_hashes is not None:
                audit_hashes[(month, symbol)] = pins[(month, symbol)]
    result.sort(key=lambda event: (event.time, event.symbol))
    return tuple(result)
