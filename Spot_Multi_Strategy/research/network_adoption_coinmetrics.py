"""Coin Metrics Community inputs for the frozen network-adoption study.

Only the two preregistered network metrics are requested.  The Community
endpoint serves today's historical values, not verified point-in-time
vintages; even a completion-time check cannot upgrade the evidence grade.
No API response is written to disk.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Mapping
from urllib.parse import parse_qs, urlencode, urlsplit

import requests


API_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
SYMBOL_TO_ASSET = {
    "BATUSDT": "bat",
    "CVCUSDT": "cvc",
    "KNCUSDT": "knc",
    "MKRUSDT": "mkr",
    "OMGUSDT": "omg",
    "RENUSDT": "ren",
    "ZRXUSDT": "zrx",
}
METRICS = "AdrActCnt,AssetEODCompletionTime"
_DAY_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T00:00:00(?:\.0{1,9})?Z\Z")


class NetworkMetricError(RuntimeError):
    """The source cannot be used without guessing about dates or values."""


def _daily_label(value: object) -> date:
    if not isinstance(value, str):
        raise NetworkMetricError("Coin Metrics time is not a string")
    match = _DAY_TIME_RE.fullmatch(value)
    if match is None:
        raise NetworkMetricError(f"non-midnight UTC daily metric label: {value!r}")
    try:
        return date.fromisoformat(match.group(1))
    except ValueError as exc:
        raise NetworkMetricError(f"invalid metric date: {value!r}") from exc


def _address_count(value: object) -> float:
    if not isinstance(value, str):
        raise NetworkMetricError("AdrActCnt must be a numeric string")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise NetworkMetricError(f"invalid AdrActCnt: {value!r}") from exc
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise NetworkMetricError(f"AdrActCnt is not a finite nonnegative count: {value!r}")
    result = float(number)
    if not math.isfinite(result):
        raise NetworkMetricError("AdrActCnt exceeds finite floating-point range")
    return result


def _completion_time(value: object, label: date) -> datetime:
    if not isinstance(value, str):
        raise NetworkMetricError("AssetEODCompletionTime must be EPOCH seconds string")
    try:
        seconds = Decimal(value)
    except InvalidOperation as exc:
        raise NetworkMetricError(f"invalid AssetEODCompletionTime: {value!r}") from exc
    if not seconds.is_finite() or seconds != seconds.to_integral_value():
        raise NetworkMetricError(f"completion time is not an integer second: {value!r}")
    try:
        completed = datetime.fromtimestamp(int(seconds), timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise NetworkMetricError(f"completion time is outside datetime range: {value!r}") from exc
    if completed < datetime.combine(label, time.min, timezone.utc):
        raise NetworkMetricError("completion precedes its daily metric label")
    return completed


def _validate_next_url(url: object, asset: str, first_params: Mapping[str, str]) -> str:
    if not isinstance(url, str) or not url:
        raise NetworkMetricError("missing or invalid next_page_url")
    parsed = urlsplit(url)
    base = urlsplit(API_URL)
    if (parsed.scheme, parsed.netloc, parsed.path) != (base.scheme, base.netloc, base.path):
        raise NetworkMetricError("next_page_url leaves the fixed Community endpoint")
    if parsed.fragment or parsed.username or parsed.password:
        raise NetworkMetricError("unexpected fragment or credentials in next_page_url")
    params = parse_qs(parsed.query, keep_blank_values=True)
    for key in ("assets", "metrics", "frequency", "start_time", "end_time", "paging_from"):
        if params.get(key) != [first_params[key]]:
            raise NetworkMetricError(f"next_page_url changed {key} for {asset}")
    if not params.get("next_page_token", [""])[0]:
        raise NetworkMetricError("next_page_url lacks continuation token")
    return url


def _fetch_one_asset(
    asset: str,
    start: date,
    end_exclusive: date,
    opener: requests.Session,
    *,
    max_pages: int,
) -> dict[date, tuple[float, datetime]]:
    params = {
        "assets": asset,
        "metrics": METRICS,
        "frequency": "1d",
        "start_time": start.isoformat(),
        "end_time": (end_exclusive - timedelta(days=1)).isoformat(),
        "page_size": "10000",
        "paging_from": "start",
    }
    url = f"{API_URL}?{urlencode(params)}"
    seen_urls: set[str] = set()
    points: dict[date, tuple[float, datetime]] = {}
    last_label: date | None = None
    saw_rows = False
    for _ in range(max_pages):
        if url in seen_urls:
            raise NetworkMetricError(f"repeated pagination URL for {asset}")
        seen_urls.add(url)
        try:
            response = opener.get(url, timeout=(10, 45), allow_redirects=False)
            if response.status_code != 200:
                raise NetworkMetricError(f"Coin Metrics HTTP {response.status_code} for {asset}")
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise NetworkMetricError(f"Coin Metrics response failed for {asset}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise NetworkMetricError(f"Coin Metrics JSON schema changed for {asset}")
        rows = payload["data"]
        if not rows and payload.get("next_page_url"):
            raise NetworkMetricError(f"empty nonterminal page for {asset}")
        for row in rows:
            if not isinstance(row, dict) or row.get("asset") != asset:
                raise NetworkMetricError(f"unexpected asset row in {asset} page")
            label = _daily_label(row.get("time"))
            if label < start or label >= end_exclusive:
                raise NetworkMetricError(f"out-of-window row for {asset}: {label}")
            if last_label is not None and label <= last_label:
                raise NetworkMetricError(f"non-increasing or duplicate day for {asset}: {label}")
            last_label = label
            saw_rows = True
            addresses = row.get("AdrActCnt")
            completion = row.get("AssetEODCompletionTime")
            if addresses is None or completion is None:
                # Coin Metrics documents null as missing and absent as unsupported.
                # Both make that day ineligible; they are never forward-filled.
                continue
            points[label] = (_address_count(addresses), _completion_time(completion, label))
        next_url = payload.get("next_page_url")
        next_token = payload.get("next_page_token")
        if next_url is None:
            if next_token not in (None, ""):
                raise NetworkMetricError(f"pagination token without URL for {asset}")
            if not saw_rows or not points:
                raise NetworkMetricError(f"no supported metrics in date range for {asset}")
            return points
        url = _validate_next_url(next_url, asset, params)
    raise NetworkMetricError(f"Coin Metrics exceeded {max_pages} pages for {asset}")


def fetch_network_metrics(
    start: date,
    end_exclusive: date,
    *,
    opener: requests.Session | None = None,
    max_pages: int = 10,
) -> dict[str, dict[date, tuple[float, datetime]]]:
    """Fetch fixed-asset daily address counts and completion times in memory.

    `end_exclusive` is not queried.  Missing/null daily values remain absent;
    malformed non-null values, HTTP errors, duplicate days and pagination
    anomalies fail closed.  The returned completion timestamps only test the
    registered publication lag, *not* historical-vintage integrity.
    """
    if type(start) is not date or type(end_exclusive) is not date or start >= end_exclusive:
        raise ValueError("start and end_exclusive must be ordered dates")
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    if opener is None:
        with requests.Session() as session:
            return {
                symbol: _fetch_one_asset(asset, start, end_exclusive, session, max_pages=max_pages)
                for symbol, asset in SYMBOL_TO_ASSET.items()
            }
    return {
        symbol: _fetch_one_asset(asset, start, end_exclusive, opener, max_pages=max_pages)
        for symbol, asset in SYMBOL_TO_ASSET.items()
    }


def address_growth_score(
    points: Mapping[date, tuple[float, datetime]], decision_time: datetime
) -> float | None:
    """Frozen Monday 08:00 UTC signal; None means no eligible 35-day window."""
    if decision_time.tzinfo is None or decision_time.utcoffset() != timedelta(0):
        raise ValueError("decision_time must be timezone-aware UTC")
    if (decision_time.weekday(), decision_time.hour, decision_time.minute,
            decision_time.second, decision_time.microsecond) != (0, 8, 0, 0, 0):
        raise ValueError("decision_time must be Monday 08:00:00 UTC")
    friday = decision_time.date() - timedelta(days=3)
    history: list[float] = []
    for lag in range(34, -1, -1):
        point = points.get(friday - timedelta(days=lag))
        if point is None:
            return None
        addresses, completed = point
        if (not math.isfinite(addresses) or addresses <= 0 or
                completed.tzinfo is None or completed.utcoffset() != timedelta(0) or
                completed > decision_time):
            return None
        history.append(addresses)
    previous_28 = sum(history[:28]) / 28
    recent_7 = sum(history[28:]) / 7
    if previous_28 <= 0 or recent_7 <= 0:
        return None
    return math.log(recent_7 / previous_28)
