"""Frozen U.S. Treasury 10-year TIPS real-yield source for one crypto study.

Do not invoke the real loader before the corresponding preregistration and
implementation are separately committed/pushed. Tests use injected bytes.
No strategy return, price, order or portfolio state is computed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import StringIO
from math import isfinite
from typing import Callable
from urllib.parse import urlsplit
import csv
import time

import requests


URL_ARCHIVE = "https://home.treasury.gov/system/files/276/par-real-yield-curve-rates-2003-2024.csv"
URL_YEAR = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_real_yield_curve&page&_format=csv"
)
URLS = (URL_ARCHIVE, URL_YEAR.format(year=2025), URL_YEAR.format(year=2026))
HEADER = ("Date", "5 YR", "7 YR", "10 YR", "20 YR", "30 YR")
FIRST_DAY = date(2020, 11, 1)
LAST_DAY = date(2026, 9, 30)
MAX_BYTES = 1_000_000


class TreasurySourceError(RuntimeError):
    """The pre-registered source is unavailable or inconsistent."""


@dataclass(frozen=True)
class YieldObservation:
    observed_date: date
    value_percent: float
    proxy_available_at: datetime


@dataclass(frozen=True)
class TreasuryBundle:
    observations: tuple[YieldObservation, ...]
    manifest: tuple[dict, ...]


def _download(url: str) -> bytes:
    for attempt in range(3):
        try:
            with requests.get(url, timeout=(10, 35), stream=True) as response:
                status = response.status_code
                if urlsplit(response.url).hostname != "home.treasury.gov":
                    raise TreasurySourceError("Treasury response redirected off official host")
                if status == 200:
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=32_768):
                        body.extend(chunk)
                        if len(body) > MAX_BYTES:
                            raise TreasurySourceError("Treasury CSV exceeds one-megabyte limit")
                    if body:
                        return bytes(body)
        except TreasurySourceError:
            raise
        except requests.RequestException as exc:
            if attempt == 2:
                raise TreasurySourceError(f"Treasury request failed for {url}") from exc
        else:
            if status != 429 and not 500 <= status <= 599:
                raise TreasurySourceError(f"Treasury HTTP {status} or empty body: {url}")
            if attempt == 2:
                raise TreasurySourceError(f"Treasury HTTP {status}: {url}")
        time.sleep((0.5, 1.0)[attempt])
    raise AssertionError("unreachable")


def _parse_date(raw: str, source_index: int) -> date:
    if not isinstance(raw, str):
        raise TreasurySourceError("Treasury row lacks a date")
    formats = ("%m/%d/%y", "%m/%d/%Y") if source_index == 0 else ("%m/%d/%Y",)
    for format_string in formats:
        try:
            return datetime.strptime(raw.strip(), format_string).date()
        except ValueError:
            continue
    raise TreasurySourceError(f"invalid Treasury date in source {source_index}")


def _window(source_index: int) -> tuple[date, date]:
    return ((FIRST_DAY, date(2024, 12, 31)),
            (date(2025, 1, 1), date(2025, 12, 31)),
            (date(2026, 1, 1), LAST_DAY))[source_index]


def _parse_source(body: bytes, source_index: int) -> tuple[tuple[YieldObservation, ...], dict]:
    if not isinstance(body, bytes) or not body or len(body) > MAX_BYTES:
        raise TreasurySourceError("empty or oversized Treasury CSV")
    try:
        content = body.decode("utf-8-sig")
    except UnicodeError as exc:
        raise TreasurySourceError("Treasury CSV is not UTF-8") from exc
    reader = csv.DictReader(StringIO(content, newline=""))
    if tuple(reader.fieldnames or ()) != HEADER:
        raise TreasurySourceError("Treasury CSV columns changed")
    start, end = _window(source_index)
    rows: list[YieldObservation] = []
    seen: set[date] = set()
    raw_count = 0
    for row in reader:
        raw_count += 1
        if None in row or any(value is None for value in row.values()):
            raise TreasurySourceError("Treasury CSV row has extra or missing fields")
        observed = _parse_date(row["Date"], source_index)
        if not start <= observed <= end:
            # The 2026 response also contains post-evaluation dates. Their
            # numerical values are deliberately not parsed or retained.
            continue
        if observed in seen:
            raise TreasurySourceError(f"duplicate Treasury date {observed}")
        seen.add(observed)
        raw = row["10 YR"]
        if not isinstance(raw, str) or not raw or raw.strip() != raw:
            raise TreasurySourceError(f"missing or malformed 10-year yield on {observed}")
        try:
            value = float(raw)
        except ValueError as exc:
            raise TreasurySourceError(f"nonnumeric 10-year yield on {observed}") from exc
        if not isfinite(value) or not -5.0 <= value <= 10.0:
            raise TreasurySourceError(f"out-of-range 10-year yield on {observed}")
        available = datetime.combine(observed + timedelta(days=2),
                                     datetime.min.time(), timezone.utc)
        rows.append(YieldObservation(observed, value, available))
    rows.sort(key=lambda row: row.observed_date)
    if not rows:
        raise TreasurySourceError(f"Treasury source {source_index} has no registered dates")
    manifest = {
        "source_index": source_index, "url": URLS[source_index],
        "http_status": 200, "bytes": len(body), "sha256": sha256(body).hexdigest(),
        "columns": list(HEADER), "source_rows": raw_count,
        "accepted_rows": len(rows),
        "first_accepted_date": rows[0].observed_date.isoformat(),
        "last_accepted_date": rows[-1].observed_date.isoformat(),
    }
    return tuple(rows), manifest


def load_and_audit(*, fetcher: Callable[[str], bytes] | None = None) -> TreasuryBundle:
    """Load only the three frozen official files and enforce date coverage."""
    get = fetcher or _download
    parts = [_parse_source(get(url), index) for index, url in enumerate(URLS)]
    observations = sorted((row for rows, _ in parts for row in rows),
                          key=lambda row: row.observed_date)
    if len({row.observed_date for row in observations}) != len(observations):
        raise TreasurySourceError("cross-source Treasury date overlap")
    counts = {year: 0 for year in range(2020, 2027)}
    for row in observations:
        counts[row.observed_date.year] += 1
    if (counts[2020] < 35 or any(counts[year] < 200 for year in range(2021, 2026))
            or counts[2026] < 175):
        raise TreasurySourceError(f"Treasury year coverage insufficient: {counts}")
    if any((right.observed_date - left.observed_date).days > 7
           for left, right in zip(observations, observations[1:])):
        raise TreasurySourceError("Treasury observation gap exceeds seven days")
    return TreasuryBundle(tuple(observations), tuple(manifest for _, manifest in parts))
