"""Checksum-verified, in-memory Binance BTC open-interest weekly source.

Only preregistered Wednesdays are read. CSV/ZIP bodies are never persisted;
published historical files are a current retrospective vintage, not PIT.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
from io import BytesIO, StringIO
from statistics import median
from typing import Callable
import csv
import re
import time
import zipfile

import pandas as pd
import requests


FIRST = pd.Timestamp("2020-10-28", tz="UTC")
LAST = pd.Timestamp("2025-12-31", tz="UTC")
WEEKS = 271
SYMBOL = "BTCUSDT"
BASE = "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/"
HEADER = (
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
)
MIN_VALID_SLOTS = 231
MAX_COMPRESSED = 2_000_000
MAX_UNCOMPRESSED = 8_000_000
_CHECKSUM = re.compile(r"([0-9a-fA-F]{64})\s+\*?([^\s]+)\s*\Z")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:00\Z")


class OiDataError(RuntimeError):
    """Registered OI source, schema or coverage failure."""


@dataclass(frozen=True)
class OiObservation:
    release_date: pd.Timestamp
    observed_date: pd.Timestamp
    value: Decimal


@dataclass(frozen=True)
class OiBundle:
    observations: tuple[OiObservation, ...]
    manifest: dict


def _download(url: str) -> bytes:
    limit = 200 if url.endswith(".CHECKSUM") else MAX_COMPRESSED
    for attempt in range(4):
        try:
            with requests.get(url, timeout=(10, 35), stream=True) as response:
                if response.status_code == 200:
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        body.extend(chunk)
                        if len(body) > limit:
                            raise OiDataError(f"oversized source body for {url}")
                    if body:
                        return bytes(body)
                status = response.status_code
        except OiDataError:
            raise
        except requests.RequestException as exc:
            if attempt == 3:
                raise OiDataError(f"transport failure for {url}") from exc
        else:
            if status != 429 and not 500 <= status <= 599:
                raise OiDataError(f"HTTP {status} or empty body for {url}")
            if attempt == 3:
                raise OiDataError(f"HTTP {status} for {url}")
        time.sleep((0.5, 1.0, 2.0)[attempt])
    raise AssertionError("unreachable")


def _numeric(raw: str, *, day: str) -> Decimal:
    if not raw or len(raw) > 128 or raw.strip() != raw:
        raise OiDataError(f"{day}: malformed OI literal")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise OiDataError(f"{day}: nondecimal OI") from exc
    if not value.is_finite() or abs(value.as_tuple().exponent) > 64:
        raise OiDataError(f"{day}: nonfinite or over-precision OI")
    return value


def _parse_day(day: pd.Timestamp, zip_body: bytes, checksum_body: bytes) -> tuple[OiObservation, dict]:
    label = day.date().isoformat()
    filename = f"{SYMBOL}-metrics-{label}.zip"
    url = BASE + filename
    if len(zip_body) > MAX_COMPRESSED or len(checksum_body) > 200:
        raise OiDataError(f"{label}: oversized source body")
    try:
        checksum_text = checksum_body.decode("ascii")
    except UnicodeError as exc:
        raise OiDataError(f"{label}: non-ASCII checksum") from exc
    match = _CHECKSUM.fullmatch(checksum_text)
    if match is None or match.group(2) != filename:
        raise OiDataError(f"{label}: checksum line not for exact archive")
    digest = sha256(zip_body).hexdigest()
    if digest.lower() != match.group(1).lower():
        raise OiDataError(f"{label}: SHA-256 mismatch")
    try:
        with zipfile.ZipFile(BytesIO(zip_body)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != filename[:-4] + ".csv"
                    or members[0].file_size > MAX_UNCOMPRESSED
                    or members[0].is_dir()):
                raise OiDataError(f"{label}: invalid ZIP member")
            raw = archive.read(members[0])
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise OiDataError(f"{label}: corrupt ZIP") from exc
    if len(raw) > MAX_UNCOMPRESSED:
        raise OiDataError(f"{label}: oversized CSV")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise OiDataError(f"{label}: invalid UTF-8 CSV") from exc
    reader = csv.reader(StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise OiDataError(f"{label}: missing CSV header") from exc
    if tuple(header) != HEADER:
        raise OiDataError(f"{label}: CSV schema changed")
    slots: dict[str, tuple[str, ...]] = {}
    row_count = duplicate_count = 0
    try:
        for row in reader:
            row_count += 1
            if len(row) != 8 or row[1] != SYMBOL or not _TIME.fullmatch(row[0]):
                raise OiDataError(f"{label}: malformed metrics row")
            try:
                stamp = pd.Timestamp(row[0], tz="UTC")
            except (TypeError, ValueError, OverflowError) as exc:
                raise OiDataError(f"{label}: invalid metrics timestamp") from exc
            if (stamp.date() != day.date() or stamp.minute % 5 != 0
                    or stamp.second != 0 or stamp.microsecond != 0):
                raise OiDataError(f"{label}: timestamp off expected five-minute grid")
            earlier = slots.get(row[0])
            if earlier is None:
                slots[row[0]] = tuple(row)
            elif earlier != tuple(row):
                raise OiDataError(f"{label}: conflicting duplicate metrics row")
            else:
                duplicate_count += 1
    except csv.Error as exc:
        raise OiDataError(f"{label}: malformed CSV quoting") from exc
    if row_count > 2000 or len(slots) > 288:
        raise OiDataError(f"{label}: excessive rows or timestamps")
    valid: list[Decimal] = []
    for row in slots.values():
        value = _numeric(row[2], day=label)
        if value > 0:
            valid.append(value)
    if len(valid) < MIN_VALID_SLOTS:
        raise OiDataError(f"{label}: fewer than 231 valid five-minute OI slots")
    with localcontext() as context:
        context.prec = 80
        daily_median = median(valid)
    if not isinstance(daily_median, Decimal) or not daily_median.is_finite() or daily_median <= 0:
        raise OiDataError(f"{label}: invalid daily OI median")
    return OiObservation(day + pd.Timedelta(days=1), day, daily_median), {
        "date": label, "url": url, "checksum_url": url + ".CHECKSUM",
        "sha256": digest, "zip_bytes": len(zip_body), "csv_bytes": len(raw),
        "rows": row_count, "duplicate_identical_rows": duplicate_count,
        "distinct_slots": len(slots), "valid_slots": len(valid),
        "invalid_nonpositive_slots": len(slots) - len(valid),
    }


def _one(day: pd.Timestamp, fetcher: Callable[[str], bytes]) -> tuple[OiObservation, dict]:
    label = day.date().isoformat()
    url = BASE + f"{SYMBOL}-metrics-{label}.zip"
    checksum = fetcher(url + ".CHECKSUM")
    archive = fetcher(url)
    if not isinstance(checksum, bytes) or not isinstance(archive, bytes):
        raise OiDataError(f"{label}: nonbyte response")
    return _parse_day(day, archive, checksum)


def load_and_audit(*, fetcher: Callable[[str], bytes] | None = None,
                   workers: int = 8) -> OiBundle:
    """Fetch only the frozen 271 days, without retaining any raw source body."""
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("worker count outside bounded range")
    days = pd.date_range(FIRST, LAST, freq="W-WED", tz="UTC")
    if len(days) != WEEKS:
        raise OiDataError("registered Wednesday grid changed")
    source = fetcher or _download
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda day: _one(day, source), days))
    observations = tuple(item[0] for item in results)
    details = [item[1] for item in results]
    return OiBundle(observations, {
        "source": BASE, "asset": SYMBOL, "field": "sum_open_interest",
        "aggregation": "median of unique positive five-minute slots within each Wednesday UTC day",
        "first_day": FIRST.date().isoformat(), "last_day": LAST.date().isoformat(),
        "weekly_observations": len(observations), "minimum_valid_slots": MIN_VALID_SLOTS,
        "total_identical_duplicate_rows": sum(x["duplicate_identical_rows"] for x in details),
        "days_with_identical_duplicates": sum(x["duplicate_identical_rows"] > 0 for x in details),
        "days_with_nonpositive_slots": sum(x["invalid_nonpositive_slots"] > 0 for x in details),
        "historical_as_of_vintages_verified": False,
        "release_proxy": "Wednesday file assumed visible Thursday 00:00 UTC; unverified",
        "archives": details,
    })
