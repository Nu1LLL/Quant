"""One-shot, current-vintage BTC MVRV reader; no source access at import time.

Historical publication times and as-of vintages are unavailable. The D+1
release date is an explicitly unverified retrospective proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from statistics import median
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlsplit
import json
import math
import re
import time

import pandas as pd
import requests


BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
PARAMS = {
    "assets": "btc", "metrics": "CapMVRVCur", "frequency": "1d",
    "start_time": "2019-10-30", "end_time": "2025-12-31",
    "page_size": "10000", "paging_from": "start",
}
FIRST_URL = BASE + "?" + urlencode(PARAMS)
FIRST = pd.Timestamp("2019-10-30", tz="UTC")
LAST = pd.Timestamp("2025-12-31", tz="UTC")
FIRST_WED = pd.Timestamp("2020-10-28", tz="UTC")
DAY_COUNT = 2255
WEEK_COUNT = 271
RELEASE_LAG = pd.Timedelta(days=1)
_MIDNIGHT = re.compile(r"\d{4}-\d{2}-\d{2}T00:00:00(?:\.0{1,9})?Z\Z")


class MvrvDataError(RuntimeError):
    """The frozen source, calendar or signal eligibility failed."""


@dataclass(frozen=True)
class DirectionObservation:
    release_date: pd.Timestamp
    observed_date: pd.Timestamp
    direction: int


@dataclass(frozen=True)
class MvrvBundle:
    observations: tuple[DirectionObservation, ...]
    manifest: dict


def _fetch(url: str) -> tuple[bytes, int]:
    for attempt in range(1, 5):
        try:
            response = requests.get(url, timeout=(10, 45))
        except requests.RequestException as exc:
            if attempt == 4:
                raise MvrvDataError("Coin Metrics transport failed") from exc
        else:
            if response.status_code == 200 and response.content:
                return response.content, attempt
            if response.status_code != 429 and not 500 <= response.status_code <= 599:
                raise MvrvDataError(f"Coin Metrics HTTP {response.status_code}")
            if attempt == 4:
                raise MvrvDataError(f"Coin Metrics HTTP {response.status_code}")
        time.sleep((0.5, 1.0, 2.0)[attempt - 1])
    raise AssertionError("unreachable")


def _url(url: object, *, first: bool = False) -> str:
    if not isinstance(url, str) or (first and url != FIRST_URL):
        raise MvrvDataError("Coin Metrics URL changed")
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc != "community-api.coinmetrics.io"
            or parts.path != "/v4/timeseries/asset-metrics" or parts.fragment
            or parts.username or parts.password):
        raise MvrvDataError("Coin Metrics pagination left the frozen endpoint")
    try:
        query = parse_qs(parts.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise MvrvDataError("invalid pagination query") from exc
    required = set(PARAMS) | (set() if first else {"next_page_token"})
    if (set(query) != required or any(query.get(k) != [v] for k, v in PARAMS.items())
            or (not first and query["next_page_token"] == [""])):
        raise MvrvDataError("pagination modified frozen query")
    return url


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise MvrvDataError("duplicate JSON key")
        result[key] = value
    return result


def _value(raw: object) -> Decimal:
    if not isinstance(raw, str) or not raw or len(raw) > 128 or raw.strip() != raw:
        raise MvrvDataError("MVRV must be a bounded decimal string")
    try:
        value = Decimal(raw)
        approximate = float(value)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise MvrvDataError("invalid MVRV decimal") from exc
    if not value.is_finite() or value <= 0 or not math.isfinite(approximate):
        raise MvrvDataError("nonpositive or nonfinite MVRV")
    if abs(value.as_tuple().exponent) > 64:
        raise MvrvDataError("MVRV decimal scale exceeds bound")
    return value


def _day(raw: object) -> pd.Timestamp:
    if not isinstance(raw, str) or not _MIDNIGHT.fullmatch(raw):
        raise MvrvDataError("MVRV date is not UTC midnight")
    try:
        return pd.Timestamp(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MvrvDataError("MVRV date invalid") from exc


def directions_from_days(days: dict[pd.Timestamp, Decimal]) -> tuple[DirectionObservation, ...]:
    """Apply the frozen 365-day median without access to later days."""
    grid = pd.date_range(FIRST, LAST, freq="D", tz="UTC")
    if len(grid) != DAY_COUNT or len(days) != DAY_COUNT or set(days) != set(grid):
        raise MvrvDataError("2,255-day MVRV calendar incomplete")
    if any(not isinstance(v, Decimal) or not v.is_finite() or v <= 0 for v in days.values()):
        raise MvrvDataError("invalid MVRV value in daily grid")
    wednesdays = pd.date_range(FIRST_WED, LAST, freq="W-WED", tz="UTC")
    if len(wednesdays) != WEEK_COUNT:
        raise MvrvDataError("271-week MVRV calendar changed")
    output = []
    for wed in wednesdays:
        lookback = [days[wed - pd.Timedelta(days=offset)] for offset in range(364, -1, -1)]
        midpoint = median(lookback)
        direction = (midpoint > days[wed]) - (midpoint < days[wed])
        output.append(DirectionObservation(wed + RELEASE_LAG, wed, direction))
    return tuple(output)


def load_and_audit(*, fetcher: Callable[[str], tuple[bytes, int]] | None = None) -> MvrvBundle:
    """Fetch the specified current vintage only after prereg and code are pushed."""
    fetch = fetcher or _fetch
    url: str | None = _url(FIRST_URL, first=True)
    seen: set[str] = set()
    days: dict[pd.Timestamp, Decimal] = {}
    pages = []
    previous = None
    while url is not None:
        if url in seen or len(pages) >= 100:
            raise MvrvDataError("pagination loop or too many pages")
        seen.add(url)
        try:
            body, attempts = fetch(url)
        except MvrvDataError:
            raise
        except Exception as exc:
            raise MvrvDataError("Coin Metrics fetcher failed") from exc
        if not isinstance(body, bytes) or not body or not isinstance(attempts, int) or attempts < 1:
            raise MvrvDataError("invalid Coin Metrics response")
        try:
            parsed = json.loads(body.decode("utf-8-sig"), object_pairs_hook=_unique_pairs)
        except (ValueError, UnicodeError) as exc:
            raise MvrvDataError("invalid Coin Metrics JSON") from exc
        rows = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(rows, list) or not rows:
            raise MvrvDataError("missing or empty Coin Metrics data")
        page_first = page_last = None
        for row in rows:
            if not isinstance(row, dict) or row.get("asset") != "btc" or "CapMVRVCur" not in row:
                raise MvrvDataError("wrong asset or missing MVRV metric")
            day = _day(row.get("time"))
            if day < FIRST or day > LAST or (previous is not None and day <= previous):
                raise MvrvDataError("MVRV dates out of range, duplicate or unordered")
            days[day] = _value(row["CapMVRVCur"])
            previous = day
            page_first = page_first or day
            page_last = day
        pages.append({
            "url": url, "http_status": 200, "attempts": attempts,
            "bytes": len(body), "sha256": sha256(body).hexdigest(),
            "rows": len(rows), "first_day": page_first.date().isoformat(),
            "last_day": page_last.date().isoformat(),
        })
        next_url = parsed.get("next_page_url")
        token = parsed.get("next_page_token")
        if next_url is None:
            if token is not None:
                raise MvrvDataError("pagination token without URL")
            url = None
        else:
            url = _url(next_url)
            if token is not None and parse_qs(urlsplit(url).query)["next_page_token"] != [token]:
                raise MvrvDataError("pagination token mismatch")
    observations = directions_from_days(days)
    return MvrvBundle(observations, {
        "source": BASE, "asset": "btc", "metric": "CapMVRVCur", "frequency": "1d",
        "first_day": FIRST.date().isoformat(), "last_day": LAST.date().isoformat(),
        "daily_rows": len(days), "weekly_observations": len(observations),
        "release_proxy": "D+1 day 00:00 UTC, unverified current vintage",
        "historical_as_of_vintages_verified": False, "pages": pages,
    })
