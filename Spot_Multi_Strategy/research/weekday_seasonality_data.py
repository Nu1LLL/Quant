"""Verified source loader for the frozen eight-coin weekday-seasonality test."""

from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import math
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

import numpy as np

from .basis_hourly_preflight import _fetch
from .crypto_numeric_preflight import _row_flags


ROOT = Path(__file__).parent
SYMBOLS = ("ADAUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "ETHUSDT",
           "LINKUSDT", "THETAUSDT", "XMRUSDT")
MONTHS = tuple(f"{year}-{month:02d}" for year in range(2021, 2027)
               for month in range(1, 13)
               if (2021, 1) <= (year, month) <= (2026, 9))
BASE_MS = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
PRICE_FIELDS = ("month", "market", "symbol", "dataset", "sha256", "bars")
FUNDING_FIELDS = ("month", "market", "symbol", "dataset", "sha256", "events", "non_hour_events")
PRICE_MANIFESTS = (
    ROOT / "preregistrations/2026-11-07_leadlag_5m_timestamp_manifest.csv",
    ROOT / "preregistrations/2026-11-07_2026_5m_source_manifest.csv",
)
FUNDING_MANIFESTS = (
    ROOT / "preregistrations/2026-11-07_leadlag_funding_timestamp_manifest.csv",
    ROOT / "preregistrations/2026-11-07_2026_funding_source_manifest.csv",
)


class WeekdayDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WeekdayTape:
    opening: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote_volume: np.ndarray
    funding: dict[int, tuple[tuple[int, float], ...]]
    price_hashes: dict[tuple[str, str], str]
    funding_hashes: dict[tuple[str, str], str]


def _pins(paths: tuple[Path, ...], fields: tuple[str, ...], kind: str
          ) -> dict[tuple[str, str], dict[str, str]]:
    pins = {}
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, strict=True)
                if tuple(reader.fieldnames or ()) != fields:
                    raise WeekdayDataError(f"{kind} manifest schema changed: {path.name}")
                rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise WeekdayDataError(f"{kind} manifest unavailable: {path.name}") from exc
        for row in rows:
            month, symbol = row["month"], row["symbol"]
            if month not in MONTHS or symbol not in SYMBOLS:
                continue
            key = (month, symbol)
            if (key in pins or row["market"] != "um" or row["dataset"] != kind
                    or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
                raise WeekdayDataError(f"{kind} source pin malformed/duplicated: {key}")
            try:
                count = int(row["bars" if kind == "5m" else "events"])
            except ValueError as exc:
                raise WeekdayDataError(f"{kind} source pin count malformed: {key}") from exc
            y, m = map(int, month.split("-"))
            if ((kind == "5m" and count != monthrange(y, m)[1] * 288)
                    or (kind == "fundingRate" and (count <= 0 or row["non_hour_events"] != "0"))):
                raise WeekdayDataError(f"{kind} source pin count changed: {key}")
            pins[key] = row
    if len(pins) != len(MONTHS) * len(SYMBOLS):
        raise WeekdayDataError(f"{kind} complete source pin grid missing")
    return pins


def _verified_zip(symbol: str, month: str, kind: str, pin: str) -> ZipFile:
    if kind == "5m":
        name = f"{symbol}-5m-{month}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/5m/{name}"
        cap, member_cap = 5_000_000, 10_000_000
    elif kind == "fundingRate":
        name = f"{symbol}-fundingRate-{month}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{name}"
        cap, member_cap = 500_000, 1_000_000
    else:
        raise ValueError("unknown source kind")
    statement = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise WeekdayDataError(f"official checksum malformed: {symbol} {month} {kind}")
    body = _fetch(url, limit=cap)
    digest = sha256(body).hexdigest()
    if digest != pin or digest != match.group(1).lower():
        raise WeekdayDataError(f"source ZIP differs from pin/checksum: {symbol} {month} {kind}")
    try:
        archive = ZipFile(io.BytesIO(body))
        members = archive.infolist()
        if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                or members[0].file_size > member_cap or members[0].flag_bits & 1):
            archive.close()
            raise WeekdayDataError(f"unexpected ZIP member: {symbol} {month} {kind}")
        return archive
    except BadZipFile as exc:
        raise WeekdayDataError(f"bad ZIP: {symbol} {month} {kind}") from exc


def _price_month(symbol: str, month: str, pin: dict[str, str]) -> np.ndarray:
    year, number = map(int, month.split("-"))
    required = monthrange(year, number)[1] * 288
    expected = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp() * 1000)
    values = np.empty((required, 5), dtype=np.float64)
    count = 0
    try:
        with _verified_zip(symbol, month, "5m", pin["sha256"]) as archive:
            with archive.open(archive.infolist()[0]) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                for fields in reader:
                    if count == 0 and fields and fields[0] == "open_time":
                        continue
                    if (count >= required or len(fields) != 12
                            or fields[0] != str(expected)
                            or fields[6] != str(expected + 300_000 - 1)):
                        raise WeekdayDataError(f"five-minute grid changed: {symbol} {month} row {count}")
                    flags = _row_flags(fields)
                    if flags:
                        raise WeekdayDataError(f"source numeric invariant {flags}: {symbol} {month} row {count}")
                    values[count] = tuple(float(fields[index]) for index in (1, 2, 3, 4, 7))
                    count += 1
                    expected += 300_000
    except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
        raise WeekdayDataError(f"invalid price CSV: {symbol} {month}") from exc
    if count != required or count != int(pin["bars"]):
        raise WeekdayDataError(f"price row count changed: {symbol} {month}")
    return values


def _funding_month(symbol: str, month: str, pin: dict[str, str]
                   ) -> tuple[tuple[int, float], ...]:
    year, number = map(int, month.split("-"))
    first = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp() * 1000)
    last = int(datetime(year + (number == 12), number % 12 + 1, 1,
                        tzinfo=timezone.utc).timestamp() * 1000)
    prior = -1
    events = []
    try:
        with _verified_zip(symbol, month, "fundingRate", pin["sha256"]) as archive:
            with archive.open(archive.infolist()[0]) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                if tuple(next(reader, ())) != ("calc_time", "funding_interval_hours", "last_funding_rate"):
                    raise WeekdayDataError(f"funding schema changed: {symbol} {month}")
                for fields in reader:
                    if len(fields) != 3:
                        raise WeekdayDataError(f"funding row shape changed: {symbol} {month}")
                    try:
                        stamp = int(fields[0])
                        interval = float(fields[1])
                        rate = float(fields[2])
                    except (ValueError, OverflowError) as exc:
                        raise WeekdayDataError(f"funding numeric parse: {symbol} {month}") from exc
                    if (not first <= stamp < last or stamp <= prior
                            or interval not in (1.0, 2.0, 4.0, 8.0)
                            or stamp % 3_600_000 >= 1_000
                            or not math.isfinite(rate) or abs(rate) > 0.1):
                        raise WeekdayDataError(f"invalid funding event: {symbol} {month}")
                    events.append((stamp, rate))
                    prior = stamp
    except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
        raise WeekdayDataError(f"invalid funding CSV: {symbol} {month}") from exc
    if len(events) != int(pin["events"]):
        raise WeekdayDataError(f"funding event count changed: {symbol} {month}")
    return tuple(events)


def _funding_bar_index(stamp: int, total: int) -> int:
    """Place sub-second source timestamps at their UTC-hour settlement bar."""
    offset_ms = stamp % 3_600_000
    if offset_ms >= 1_000:
        raise WeekdayDataError("funding timestamp more than 999 ms after UTC hour")
    settled_hour = stamp - offset_ms
    delta = settled_hour - BASE_MS
    if delta % 300_000 or not 0 <= delta // 300_000 < total:
        raise WeekdayDataError("funding event outside price tape")
    return delta // 300_000


def load_tape() -> WeekdayTape:
    """Validate *all* consumed archives before a simulator sees any values."""
    price_pins = _pins(PRICE_MANIFESTS, PRICE_FIELDS, "5m")
    funding_pins = _pins(FUNDING_MANIFESTS, FUNDING_FIELDS, "fundingRate")
    total = sum(monthrange(*map(int, month.split("-")))[1] * 288 for month in MONTHS)
    arrays = [np.empty((total, len(SYMBOLS)), dtype=np.float64) for _ in range(5)]
    offset = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for month in MONTHS:
            jobs = {symbol: pool.submit(_price_month, symbol, month,
                                        price_pins[(month, symbol)]) for symbol in SYMBOLS}
            required = monthrange(*map(int, month.split("-")))[1] * 288
            for col, symbol in enumerate(SYMBOLS):
                values = jobs[symbol].result()
                if values.shape != (required, 5):
                    raise WeekdayDataError(f"price month shape changed: {symbol} {month}")
                for k in range(5):
                    arrays[k][offset:offset + required, col] = values[:, k]
            offset += required
    if offset != total:
        raise WeekdayDataError("price tape length changed")
    for array in arrays:
        array.setflags(write=False)

    funding: dict[int, list[tuple[int, float]]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for month in MONTHS:
            jobs = {symbol: pool.submit(_funding_month, symbol, month,
                                        funding_pins[(month, symbol)]) for symbol in SYMBOLS}
            for col, symbol in enumerate(SYMBOLS):
                for stamp, rate in jobs[symbol].result():
                    try:
                        index = _funding_bar_index(stamp, total)
                    except WeekdayDataError as exc:
                        raise WeekdayDataError(f"{exc}: {symbol} {month}") from exc
                    funding.setdefault(index, []).append((col, rate))
    return WeekdayTape(*arrays,
                       {index: tuple(events) for index, events in sorted(funding.items())},
                       {key: row["sha256"] for key, row in price_pins.items()},
                       {key: row["sha256"] for key, row in funding_pins.items()})
