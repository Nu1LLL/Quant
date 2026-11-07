"""Pinned, whole-archive data reader for the registered ten-coin experiment.

It verifies both committed and freshly published Binance checksums and the
complete grid before returning any price array to the signal simulator. Raw
archive bytes are held in memory only; funding-rate values are not read.
"""

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
from typing import MutableMapping
from zipfile import BadZipFile, ZipFile

import numpy as np

from .basis_hourly_preflight import _fetch
from .leadlag_5m_preflight import SYMBOLS as SCREEN_SYMBOLS


UTC = timezone.utc
SYMBOLS = tuple(sorted(SCREEN_SYMBOLS))
PRICE_MANIFEST = Path(__file__).parent / "preregistrations/2026-11-07_leadlag_5m_timestamp_manifest.csv"
FUNDING_MANIFEST = Path(__file__).parent / "preregistrations/2026-11-07_leadlag_funding_timestamp_manifest.csv"
PRICE_FIELDS = ("month", "market", "symbol", "dataset", "sha256", "bars")
FUNDING_FIELDS = ("month", "market", "symbol", "dataset", "sha256", "events", "non_hour_events")


class LeadLagDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PriceTape:
    opening: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    first_open_ms: int

    def __post_init__(self) -> None:
        shapes = {x.shape for x in (self.opening, self.high, self.low, self.close)}
        if len(shapes) != 1 or self.opening.ndim != 2 or self.opening.shape[1] != len(SYMBOLS):
            raise LeadLagDataError("price tape arrays have mismatched shapes")


def _months(first: tuple[int, int], last: tuple[int, int]) -> tuple[str, ...]:
    return tuple(f"{y}-{m:02d}" for y in range(first[0], last[0] + 1)
                 for m in range(1, 13) if first <= (y, m) <= last)


def _pins(path: Path, *, kind: str) -> dict[tuple[str, str], dict[str, str]]:
    if kind == "5m":
        months = _months((2020, 12), (2025, 12))
        fields = PRICE_FIELDS
        count_key = "bars"
    elif kind == "fundingRate":
        months = _months((2021, 1), (2025, 12))
        fields = FUNDING_FIELDS
        count_key = "events"
    else:
        raise ValueError("unknown archive kind")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != fields:
                raise LeadLagDataError(f"{kind} manifest schema changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise LeadLagDataError(f"{kind} manifest unavailable") from exc
    if len(rows) != len(months) * len(SYMBOLS):
        raise LeadLagDataError(f"{kind} manifest row count changed")
    pins: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        month, symbol = row.get("month"), row.get("symbol")
        if (set(row) != set(fields) or month not in months or symbol not in SYMBOLS
                or row.get("market") != "um" or row.get("dataset") != kind
                or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256") or "")
                or (month, symbol) in pins):
            raise LeadLagDataError(f"{kind} manifest row changed")
        try:
            count = int(row[count_key])
            y, m = map(int, month.split("-"))
        except (ValueError, TypeError) as exc:
            raise LeadLagDataError(f"{kind} manifest count malformed") from exc
        if kind == "5m" and count != monthrange(y, m)[1] * 288:
            raise LeadLagDataError("5m manifest bar count changed")
        if kind == "fundingRate" and (count <= 0 or row["non_hour_events"] != "0"):
            raise LeadLagDataError("funding manifest event count or hour status changed")
        pins[(month, symbol)] = row
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
        raise ValueError("unknown archive kind")
    try:
        statement = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    except UnicodeError as exc:
        raise LeadLagDataError(f"non-ASCII official checksum: {symbol} {month}") from exc
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise LeadLagDataError(f"official checksum malformed: {symbol} {month}")
    body = _fetch(url, limit=cap)
    digest = sha256(body).hexdigest()
    if digest != pin or digest != match.group(1).lower():
        raise LeadLagDataError(f"ZIP digest differs from pin or official checksum: {symbol} {month}")
    try:
        archive = ZipFile(io.BytesIO(body))
        members = archive.infolist()
        if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                or members[0].flag_bits & 1 or members[0].file_size > member_cap):
            archive.close()
            raise LeadLagDataError(f"unexpected ZIP member: {symbol} {month}")
        return archive
    except BadZipFile as exc:
        raise LeadLagDataError(f"bad ZIP: {symbol} {month}") from exc


def _funding_month(symbol: str, month: str, pin: dict[str, str]) -> int:
    year, number = map(int, month.split("-"))
    first = int(datetime(year, number, 1, tzinfo=UTC).timestamp() * 1000)
    last = int(datetime(year + (number == 12), number % 12 + 1, 1,
                        tzinfo=UTC).timestamp() * 1000)
    count, prior = 0, -1
    try:
        with _verified_zip(symbol, month, "fundingRate", pin["sha256"]) as archive:
            with archive.open(archive.infolist()[0]) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                if tuple(next(reader, ())) != ("calc_time", "funding_interval_hours", "last_funding_rate"):
                    raise LeadLagDataError(f"funding schema changed: {symbol} {month}")
                for fields in reader:
                    if len(fields) != 3:
                        raise LeadLagDataError(f"funding row shape changed: {symbol} {month}")
                    try:
                        stamp = int(fields[0])
                        interval = float(fields[1])
                    except ValueError as exc:
                        raise LeadLagDataError(f"invalid funding timestamp/interval: {symbol} {month}") from exc
                    if (not first <= stamp < last or stamp <= prior
                            or interval not in (1.0, 2.0, 4.0, 8.0)
                            or stamp % 3_600_000 >= 1_000):
                        raise LeadLagDataError(f"off-hour or malformed funding event: {symbol} {month}")
                    prior = stamp
                    count += 1
    except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
        raise LeadLagDataError(f"invalid funding archive: {symbol} {month}") from exc
    if count != int(pin["events"]):
        raise LeadLagDataError(f"funding event count changed: {symbol} {month}")
    return count


def verify_funding_schedule(*, audit_hashes: MutableMapping[tuple[str, str], str] | None = None
                            ) -> int:
    pins = _pins(FUNDING_MANIFEST, kind="fundingRate")
    events = 0
    with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as pool:
        for month in _months((2021, 1), (2025, 12)):
            jobs = {symbol: pool.submit(_funding_month, symbol, month,
                                        pins[(month, symbol)]) for symbol in SYMBOLS}
            for symbol in SYMBOLS:
                events += jobs[symbol].result()
                if audit_hashes is not None:
                    audit_hashes[(month, symbol)] = pins[(month, symbol)]["sha256"]
    return events


def _price_month(symbol: str, month: str, pin: dict[str, str]) -> np.ndarray:
    year, number = map(int, month.split("-"))
    expected = int(datetime(year, number, 1, tzinfo=UTC).timestamp() * 1000)
    required = monthrange(year, number)[1] * 288
    values = np.empty((required, 4), dtype=np.float64)
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
                        raise LeadLagDataError(f"five-minute grid changed: {symbol} {month}, row={count}")
                    try:
                        o, h, l, c = (float(fields[j]) for j in (1, 2, 3, 4))
                        vol, quote, taker_base, taker_quote = (float(fields[j]) for j in (5, 7, 9, 10))
                        trades = int(fields[8])
                    except (ValueError, OverflowError) as exc:
                        raise LeadLagDataError(f"invalid price numeric field: {symbol} {month}") from exc
                    if (not all(math.isfinite(x) for x in (o, h, l, c, vol, quote,
                                                            taker_base, taker_quote))
                            or min(o, h, l, c) <= 0 or h < max(o, c, l)
                            or l > min(o, c, h) or min(vol, quote, taker_base, taker_quote) < 0
                            or taker_base > vol + 1e-9 or taker_quote > quote + 1e-6
                            or trades < 0):
                        raise LeadLagDataError(f"invalid price/volume field: {symbol} {month}")
                    values[count] = (o, h, l, c)
                    expected += 300_000
                    count += 1
    except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
        raise LeadLagDataError(f"invalid price archive: {symbol} {month}") from exc
    if count != required or count != int(pin["bars"]):
        raise LeadLagDataError(f"price bar count changed: {symbol} {month}")
    return values


def load_price_tape(*, audit_hashes: MutableMapping[tuple[str, str], str] | None = None
                    ) -> PriceTape:
    pins = _pins(PRICE_MANIFEST, kind="5m")
    months = _months((2020, 12), (2025, 12))
    total = sum(int(pins[(month, SYMBOLS[0])]["bars"]) for month in months)
    opening, high, low, close = (np.empty((total, len(SYMBOLS)), dtype=np.float64)
                                 for _ in range(4))
    offset = 0
    with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as pool:
        for month in months:
            jobs = {symbol: pool.submit(_price_month, symbol, month,
                                        pins[(month, symbol)]) for symbol in SYMBOLS}
            count = int(pins[(month, SYMBOLS[0])]["bars"])
            for col, symbol in enumerate(SYMBOLS):
                values = jobs[symbol].result()
                if len(values) != count:
                    raise LeadLagDataError(f"cross-symbol month length changed: {month}")
                opening[offset:offset + count, col] = values[:, 0]
                high[offset:offset + count, col] = values[:, 1]
                low[offset:offset + count, col] = values[:, 2]
                close[offset:offset + count, col] = values[:, 3]
                if audit_hashes is not None:
                    audit_hashes[(month, symbol)] = pins[(month, symbol)]["sha256"]
            offset += count
    if offset != total:
        raise LeadLagDataError("incomplete price tape")
    for array in (opening, high, low, close):
        array.setflags(write=False)
    first = int(datetime(2020, 12, 1, tzinfo=UTC).timestamp() * 1000)
    return PriceTape(opening, high, low, close, first)
