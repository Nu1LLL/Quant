"""Full-grid, pinned USD-M trade-size tape for the fixed three-coin study."""

from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import math
from pathlib import Path
import re

import numpy as np

from .crypto_numeric_preflight import _row_flags
from .weekday_seasonality_data import _funding_month, _verified_zip


ROOT = Path(__file__).parent
SYMBOLS = ("ADAUSDT", "BTCUSDT", "ETHUSDT")
MONTHS = tuple(f"{year}-{month:02d}" for year in range(2020, 2027)
               for month in range(1, 13)
               if (2020, 12) <= (year, month) <= (2026, 9))
FUNDING_MONTHS = tuple(month for month in MONTHS if month >= "2021-03")
BASE = datetime(2020, 12, 1, tzinfo=timezone.utc)
BASE_MS = int(BASE.timestamp() * 1000)
PRICE_FILES = (
    ROOT / "preregistrations/2026-11-07_leadlag_5m_timestamp_manifest.csv",
    ROOT / "preregistrations/2026-11-07_2026_5m_source_manifest.csv",
)
FUNDING_FILES = (
    ROOT / "preregistrations/2026-11-07_leadlag_funding_timestamp_manifest.csv",
    ROOT / "preregistrations/2026-11-07_2026_funding_source_manifest.csv",
)


class TradeSizeSourceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TradeSizeTape:
    opening: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote_volume: np.ndarray
    trade_count: np.ndarray
    funding: dict[int, tuple[tuple[int, float], ...]]
    price_hashes: dict[str, str]
    funding_hashes: dict[str, str]


def _pins(files: tuple[Path, ...], kind: str, months: tuple[str, ...]) -> dict[tuple[str, str], dict]:
    fields = (("month", "market", "symbol", "dataset", "sha256", "bars")
              if kind == "5m" else
              ("month", "market", "symbol", "dataset", "sha256", "events", "non_hour_events"))
    pins = {}
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle, strict=True)
            if tuple(rows.fieldnames or ()) != fields:
                raise TradeSizeSourceError(f"manifest schema changed: {path.name}")
            for row in rows:
                month, symbol = row["month"], row["symbol"]
                if month not in months or symbol not in SYMBOLS:
                    continue
                key = (month, symbol)
                if (key in pins or row["market"] != "um" or row["dataset"] != kind
                        or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
                    raise TradeSizeSourceError(f"manifest pin invalid: {key}")
                count = int(row["bars" if kind == "5m" else "events"])
                if kind == "5m":
                    year, number = map(int, month.split("-"))
                    if count != monthrange(year, number)[1] * 288:
                        raise TradeSizeSourceError(f"price pin count changed: {key}")
                elif count <= 0 or row["non_hour_events"] != "0":
                    raise TradeSizeSourceError(f"funding pin count changed: {key}")
                pins[key] = row
    if len(pins) != len(months) * len(SYMBOLS):
        raise TradeSizeSourceError(f"incomplete {kind} source pins")
    return pins


def _price_month(symbol: str, month: str, pin: dict) -> np.ndarray:
    year, number = map(int, month.split("-"))
    required = monthrange(year, number)[1] * 288
    expected = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp() * 1000)
    values = np.empty((required, 6), dtype=np.float64)
    count = 0
    with _verified_zip(symbol, month, "5m", pin["sha256"]) as archive:
        with archive.open(archive.infolist()[0]) as raw:
            rows = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""),
                              strict=True)
            for fields in rows:
                if count == 0 and fields and fields[0] == "open_time":
                    continue
                if (count >= required or len(fields) != 12
                        or fields[0] != str(expected)
                        or fields[6] != str(expected + 299_999)):
                    raise TradeSizeSourceError(f"five-minute grid changed: {symbol} {month} row {count}")
                flags = _row_flags(fields)
                if flags:
                    raise TradeSizeSourceError(f"source numeric invariant {flags}: {symbol} {month} row {count}")
                try:
                    trades = int(fields[8])
                except ValueError as exc:
                    raise TradeSizeSourceError(f"noninteger trade count: {symbol} {month}") from exc
                quote = float(fields[7])
                if trades < 0 or (trades == 0) != (quote == 0):
                    raise TradeSizeSourceError(f"trade-count/quote mismatch: {symbol} {month} row {count}")
                values[count] = (float(fields[1]), float(fields[2]),
                                 float(fields[3]), float(fields[4]), quote, trades)
                expected += 300_000
                count += 1
    if count != required or count != int(pin["bars"]):
        raise TradeSizeSourceError(f"price rows incomplete: {symbol} {month}")
    return values


def load_tape() -> TradeSizeTape:
    prices = _pins(PRICE_FILES, "5m", MONTHS)
    rates = _pins(FUNDING_FILES, "fundingRate", FUNDING_MONTHS)
    total = sum(monthrange(*map(int, month.split("-")))[1] * 288 for month in MONTHS)
    arrays = [np.empty((total, len(SYMBOLS)), dtype=np.float64) for _ in range(6)]
    offsets = {}
    offset = 0
    for month in MONTHS:
        offsets[month] = offset
        offset += monthrange(*map(int, month.split("-")))[1] * 288
    if offset != total:
        raise TradeSizeSourceError("source array shape changed")
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(_price_month, symbol, month, prices[(month, symbol)]):
                (month, symbol) for month in MONTHS for symbol in SYMBOLS}
        for job in as_completed(jobs):
            month, symbol = jobs[job]
            values = job.result()
            first = offsets[month]
            col = SYMBOLS.index(symbol)
            for k, array in enumerate(arrays):
                array[first:first + len(values), col] = values[:, k]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise TradeSizeSourceError("nonfinite price/count array")
    if np.any(arrays[5] != np.floor(arrays[5])):
        raise TradeSizeSourceError("noninteger trade count in tape")
    funding: dict[int, list[tuple[int, float]]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(_funding_month, symbol, month, rates[(month, symbol)]):
                (month, symbol) for month in FUNDING_MONTHS for symbol in SYMBOLS}
        for job in as_completed(jobs):
            month, symbol = jobs[job]
            col = SYMBOLS.index(symbol)
            for stamp, rate in job.result():
                delta = stamp - (stamp % 3_600_000) - BASE_MS
                if delta % 300_000 or not 0 <= delta // 300_000 < total:
                    raise TradeSizeSourceError(f"funding outside price grid: {symbol} {month}")
                funding.setdefault(delta // 300_000, []).append((col, rate))
    for array in arrays:
        array.setflags(write=False)
    return TradeSizeTape(
        *arrays,
        {index: tuple(rows) for index, rows in sorted(funding.items())},
        {f"{m}:{s}": prices[(m, s)]["sha256"] for m in MONTHS for s in SYMBOLS},
        {f"{m}:{s}": rates[(m, s)]["sha256"] for m in FUNDING_MONTHS for s in SYMBOLS},
    )
