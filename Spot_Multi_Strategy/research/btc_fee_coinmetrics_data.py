"""Strict, in-memory Coin Metrics BTC FeeTotNtv reader for the frozen experiment.

The source is a current historical vintage, not a point-in-time archive.
``release_date`` is the preregistered D+1 proxy, not a verified publication
timestamp. No raw response or daily fee value is persisted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
import json
import math
import re
import time
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlsplit

import pandas as pd
import requests


BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
PARAMETERS = {
    "assets": "btc",
    "metrics": "FeeTotNtv",
    "frequency": "1d",
    "start_time": "2020-10-01",
    "end_time": "2025-12-31",
    "page_size": "10000",
    "paging_from": "start",
}
START_URL = f"{BASE_URL}?{urlencode(PARAMETERS)}"
FIRST_DAY = pd.Timestamp("2020-10-01", tz="UTC")
LAST_DAY = pd.Timestamp("2025-12-31", tz="UTC")
FIRST_WEDNESDAY = pd.Timestamp("2020-10-28", tz="UTC")
EXPECTED_DAYS = 1918
EXPECTED_WEDNESDAYS = 271
PROXY_LAG = pd.Timedelta(days=1)
EVIDENCE_GRADE = "retrospective_current_vintage_btc_fee"
MAX_PAGES = 100
MAX_FEE_LITERAL_LENGTH = 128
MAX_FEE_FRACTIONAL_DIGITS = 64
MAX_FEE_POSITIVE_EXPONENT = 308
_UTC_MIDNIGHT = re.compile(r"\d{4}-\d{2}-\d{2}T00:00:00(?:\.0{1,9})?Z\Z")


class BtcFeeDataError(RuntimeError):
    """The registered source, transport, schema, or calendar gate failed."""


@dataclass(frozen=True)
class FeeObservation:
    release_date: pd.Timestamp
    observed_date: pd.Timestamp
    value: Decimal  # Raw 28-day rolling total R_i, not the adapter's V_i.


@dataclass(frozen=True)
class FeeBundle:
    observations: tuple[FeeObservation, ...]
    manifest: dict


@dataclass(frozen=True)
class _Payload:
    body: bytes
    status_code: int = 200
    attempts: int = 1


def _http_fetch(url: str) -> _Payload:
    """Bounded retry of transient HTTP failures, without source substitution."""
    for attempt in range(1, 5):
        try:
            response = requests.get(url, timeout=(10, 45))
        except requests.RequestException as exc:
            if attempt == 4:
                raise BtcFeeDataError(f"Coin Metrics request failed: {url}") from exc
        else:
            if response.status_code != 429 and not 500 <= response.status_code <= 599:
                return _Payload(response.content, response.status_code, attempt)
            if attempt == 4:
                return _Payload(response.content, response.status_code, attempt)
        time.sleep((0.5, 1.0, 2.0)[attempt - 1])
    raise AssertionError("unreachable Coin Metrics retry loop")


def _fetch(fetcher: Callable[[str], bytes | _Payload], url: str) -> _Payload:
    try:
        payload = fetcher(url)
    except BtcFeeDataError:
        raise
    except Exception as exc:
        raise BtcFeeDataError(f"Coin Metrics request failed: {url}") from exc
    if isinstance(payload, bytes):
        payload = _Payload(payload)
    if not isinstance(payload, _Payload) or not isinstance(payload.body, bytes):
        raise BtcFeeDataError(f"unexpected Coin Metrics response type: {url}")
    if payload.status_code != 200 or not payload.body or payload.attempts < 1:
        raise BtcFeeDataError(f"Coin Metrics HTTP {payload.status_code} or empty body: {url}")
    return payload


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise BtcFeeDataError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(body: bytes) -> dict:
    try:
        parsed = json.loads(body.decode("utf-8-sig", errors="strict"),
                            object_pairs_hook=_unique_keys)
    except BtcFeeDataError:
        raise
    except (ValueError, UnicodeError) as exc:
        raise BtcFeeDataError("invalid UTF-8 Coin Metrics JSON") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("data"), list):
        raise BtcFeeDataError("Coin Metrics page missing data array")
    return parsed


def _validate_url(url: object, *, is_first: bool = False) -> str:
    if not isinstance(url, str) or not url:
        raise BtcFeeDataError("missing or invalid next_page_url")
    if is_first and url != START_URL:
        raise BtcFeeDataError("initial Coin Metrics URL changed")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "community-api.coinmetrics.io"
            or parsed.path != "/v4/timeseries/asset-metrics" or parsed.fragment
            or parsed.username or parsed.password):
        raise BtcFeeDataError("pagination changed Coin Metrics origin or path")
    try:
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise BtcFeeDataError("invalid pagination query") from exc
    expected_keys = set(PARAMETERS) | (set() if is_first else {"next_page_token"})
    if set(query) != expected_keys or any(query.get(key) != [value]
                                               for key, value in PARAMETERS.items()):
        raise BtcFeeDataError("pagination changed frozen query parameters")
    if not is_first and query["next_page_token"] == [""]:
        raise BtcFeeDataError("empty pagination token")
    return url


def _day(raw: object) -> pd.Timestamp:
    if not isinstance(raw, str) or not _UTC_MIDNIGHT.fullmatch(raw):
        raise BtcFeeDataError("FeeTotNtv time must be UTC midnight ISO timestamp")
    try:
        return pd.Timestamp(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        raise BtcFeeDataError("FeeTotNtv time outside supported calendar") from exc


def _fee(raw: object) -> Decimal:
    # Coin Metrics serializes numeric metric values as strings.
    if (not isinstance(raw, str) or not raw or raw.strip() != raw
            or len(raw) > MAX_FEE_LITERAL_LENGTH):
        raise BtcFeeDataError("FeeTotNtv must be a numeric string in BTC units")
    try:
        value = Decimal(raw)
        as_float = float(value)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise BtcFeeDataError("FeeTotNtv is not a finite numeric value") from exc
    if not value.is_finite() or not math.isfinite(as_float) or value < 0:
        raise BtcFeeDataError("FeeTotNtv is negative or nonfinite")
    exponent = value.as_tuple().exponent
    if (exponent < -MAX_FEE_FRACTIONAL_DIGITS
            or exponent > MAX_FEE_POSITIVE_EXPONENT):
        raise BtcFeeDataError("FeeTotNtv decimal scale exceeds safe exact-sum bounds")
    return value


def _page_rows(parsed: dict) -> tuple[list[tuple[pd.Timestamp, Decimal]], dict]:
    rows = parsed["data"]
    if not rows:
        raise BtcFeeDataError("empty Coin Metrics page")
    result: list[tuple[pd.Timestamp, Decimal]] = []
    previous = None
    for row in rows:
        if not isinstance(row, dict) or row.get("asset") != "btc" or "FeeTotNtv" not in row:
            raise BtcFeeDataError("Coin Metrics page row has wrong asset or metric")
        day = _day(row.get("time"))
        if day < FIRST_DAY or day > LAST_DAY or (previous is not None and day <= previous):
            raise BtcFeeDataError("Coin Metrics page dates out of range or order")
        result.append((day, _fee(row["FeeTotNtv"])))
        previous = day
    coverage = {
        "row_count": len(result),
        "first_day": result[0][0].date().isoformat(),
        "last_day": result[-1][0].date().isoformat(),
    }
    return result, coverage


def load_and_audit(*, fetcher: Callable[[str], bytes | _Payload] | None = None) -> FeeBundle:
    """Fetch the frozen series and form raw Wednesday rolling sums in memory.

    ``fetcher`` exists for synthetic, offline testing. Calling this without a
    fetcher is a real source read and is only for the separately committed run.
    """
    fetch = fetcher or _http_fetch
    url = _validate_url(START_URL, is_first=True)
    seen_urls: set[str] = set()
    days: dict[pd.Timestamp, Decimal] = {}
    pages: list[dict] = []
    previous_day = None
    while url is not None:
        if url in seen_urls or len(pages) >= MAX_PAGES:
            raise BtcFeeDataError("Coin Metrics pagination loop or too many pages")
        seen_urls.add(url)
        payload = _fetch(fetch, url)
        parsed = _json(payload.body)
        page_rows, coverage = _page_rows(parsed)
        for day, value in page_rows:
            if previous_day is not None and day <= previous_day:
                raise BtcFeeDataError("duplicate or out-of-order day across pages")
            days[day] = value
            previous_day = day
        pages.append({
            "url": url,
            "http_status": payload.status_code,
            "http_attempts": payload.attempts,
            "sha256": sha256(payload.body).hexdigest(),
            "bytes": len(payload.body),
            **coverage,
        })
        next_url = parsed.get("next_page_url")
        next_token = parsed.get("next_page_token")
        if next_url is None:
            if next_token is not None:
                raise BtcFeeDataError("pagination token without next_page_url")
            url = None
        else:
            url = _validate_url(next_url)
            url_token = parse_qs(urlsplit(url).query)["next_page_token"][0]
            if next_token is not None and next_token != url_token:
                raise BtcFeeDataError("next_page_token disagrees with next_page_url")

    grid = pd.date_range(FIRST_DAY, LAST_DAY, freq="D", tz="UTC")
    if len(grid) != EXPECTED_DAYS or len(days) != EXPECTED_DAYS or set(days) != set(grid):
        raise BtcFeeDataError("1,918-day UTC FeeTotNtv grid incomplete")
    wednesdays = pd.date_range(FIRST_WEDNESDAY, LAST_DAY, freq="W-WED", tz="UTC")
    if len(wednesdays) != EXPECTED_WEDNESDAYS:
        raise BtcFeeDataError("expected 271 Wednesday observations")
    # Nonnegative terms need at most two carry digits when 28 are summed.
    # Integer digits + fractional scale + two therefore preserves every
    # decimal input digit, including differences smaller than 1e-28.
    integer_digits = max(1, max(value.adjusted() + 1 for value in days.values()))
    fractional_digits = max(0, max(-value.as_tuple().exponent for value in days.values()))
    precision = integer_digits + fractional_digits + 2
    observations = []
    with localcontext() as exact_context:
        exact_context.prec = precision
        for day in wednesdays:
            rolling = sum((days[day - pd.Timedelta(days=offset)]
                           for offset in range(27, -1, -1)), Decimal(0))
            if not rolling.is_finite() or not math.isfinite(float(rolling)):
                raise BtcFeeDataError("28-day BTC fee sum is nonfinite")
            observations.append(FeeObservation(day + PROXY_LAG, day, rolling))
    manifest = {
        "source": BASE_URL,
        "asset": "btc",
        "metric": "FeeTotNtv",
        "frequency": "1d",
        "evidence_grade": EVIDENCE_GRADE,
        "independent_oos": False,
        "proxy_lag_days": 1,
        "daily_row_count": len(days),
        "weekly_observation_count": len(observations),
        "first_day": FIRST_DAY.date().isoformat(),
        "last_day": LAST_DAY.date().isoformat(),
        "first_observed_date": observations[0].observed_date.date().isoformat(),
        "last_observed_date": observations[-1].observed_date.date().isoformat(),
        "pages": pages,
    }
    return FeeBundle(tuple(observations), manifest)
