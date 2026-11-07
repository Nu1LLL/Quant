"""Frozen FRED ICSA weekly source; no result data is read during import."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import StringIO
from typing import Callable
from urllib.parse import urlparse

import requests


BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
PARAMS = {"id": "ICSA", "cosd": "2020-11-01", "coed": "2026-09-30"}
FIRST = date(2020, 11, 7)
LAST = date(2026, 9, 26)
ROWS = 308
MAX_BYTES = 32_768


class ClaimsSourceError(RuntimeError):
    """The official CSV violates the preregistered source contract."""


@dataclass(frozen=True)
class ClaimsObservation:
    observed_date: date
    claims: int
    proxy_available_at: datetime


@dataclass(frozen=True)
class ClaimsBundle:
    observations: tuple[ClaimsObservation, ...]
    manifest: dict


def _download() -> tuple[bytes, str]:
    for attempt in range(3):
        try:
            with requests.get(BASE, params=PARAMS, timeout=(10, 45), stream=True) as response:
                if (response.url.split("?", 1)[0] != BASE or
                        urlparse(response.url).hostname != "fred.stlouisfed.org"):
                    raise ClaimsSourceError("FRED request redirected or URL changed")
                if response.status_code == 200:
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=8192):
                        body.extend(chunk)
                        if len(body) > MAX_BYTES:
                            raise ClaimsSourceError("FRED CSV exceeds registered size cap")
                    if body:
                        return bytes(body), response.url
                elif response.status_code not in (429, 500, 502, 503, 504):
                    raise ClaimsSourceError(f"FRED HTTP {response.status_code}")
                if attempt == 2:
                    raise ClaimsSourceError(f"FRED HTTP {response.status_code} or empty body")
        except requests.RequestException as exc:
            if attempt == 2:
                raise ClaimsSourceError("FRED request failed") from exc
        import time as clock
        clock.sleep(0.5 * 2**attempt)
    raise AssertionError("unreachable")


def _parse(body: bytes, final_url: str) -> ClaimsBundle:
    if not isinstance(body, bytes) or not body or len(body) > MAX_BYTES:
        raise ClaimsSourceError("missing or oversized FRED CSV")
    if (final_url.split("?", 1)[0] != BASE or
            urlparse(final_url).hostname != "fred.stlouisfed.org"):
        raise ClaimsSourceError("untrusted FRED CSV URL")
    try:
        decoded = body.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ClaimsSourceError("FRED CSV is not UTF-8") from exc
    try:
        rows = list(csv.reader(StringIO(decoded, newline=""), strict=True))
    except csv.Error as exc:
        raise ClaimsSourceError("malformed FRED CSV") from exc
    if not rows or rows[0] != ["DATE", "ICSA"] or len(rows) != ROWS + 1:
        raise ClaimsSourceError("FRED header or weekly row count changed")
    observations = []
    for n, row in enumerate(rows[1:]):
        if len(row) != 2:
            raise ClaimsSourceError("FRED weekly row schema changed")
        try:
            stamp = date.fromisoformat(row[0])
            amount = Decimal(row[1])
        except (ValueError, InvalidOperation) as exc:
            raise ClaimsSourceError("FRED weekly date or claim count invalid") from exc
        expected = FIRST + timedelta(days=7 * n)
        if stamp != expected or not amount.is_finite() or not 0 < amount <= 10_000_000:
            raise ClaimsSourceError("FRED weekly coverage or claim count invalid")
        if amount != int(amount):
            raise ClaimsSourceError("FRED claim count is not an integer")
        observations.append(ClaimsObservation(
            stamp, int(amount), datetime.combine(stamp + timedelta(days=9), time.min,
                                                 timezone.utc)))
    if observations[-1].observed_date != LAST:
        raise ClaimsSourceError("FRED terminal observation changed")
    manifest = {"url": final_url, "http_status": 200,
                "bytes": len(body), "sha256": sha256(body).hexdigest(),
                "source_rows": len(observations), "first_date": FIRST.isoformat(),
                "last_date": LAST.isoformat(), "unit": "persons",
                "proxy_lag_calendar_days": 9, "historical_vintage_verified": False}
    return ClaimsBundle(tuple(observations), manifest)


def load_and_audit(*, fetcher: Callable[[], tuple[bytes, str]] | None = None) -> ClaimsBundle:
    body, url = (fetcher or _download)()
    return _parse(body, url)
