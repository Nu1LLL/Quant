"""Strict, in-memory reader for the preregistered USDT+USDC supply proxy.

The source is a *current, backfilled* DefiLlama history, not a point-in-time
vintage archive.  ``release_date`` is deliberately the registered D+7 proxy,
not a claim about when an historical value actually became public.  Neither
raw responses nor daily supply values are written to disk or returned in the
manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re
import time
from typing import Callable

import pandas as pd
import requests


BASE_URL = "https://stablecoins.llama.fi"
CATALOG_URL = f"{BASE_URL}/stablecoins"
HISTORY_URLS = {
    "USDT": f"{BASE_URL}/stablecoincharts/all?stablecoin=1",
    "USDC": f"{BASE_URL}/stablecoincharts/all?stablecoin=2",
}
EXPECTED_IDS = {"1": "USDT", "2": "USDC"}
FIRST_DAY = pd.Timestamp("2020-10-28", tz="UTC")
LAST_DAY = pd.Timestamp("2025-12-31", tz="UTC")
EXPECTED_DAYS = 1891
EXPECTED_WEDNESDAYS = 271
AVAILABILITY_LAG = pd.Timedelta(days=7)
EVIDENCE_GRADE = "retrospective_backfilled_supply_proxy"


class StablecoinDataError(RuntimeError):
    """A frozen source, schema, date, or value gate failed."""


@dataclass(frozen=True)
class SupplyObservation:
    release_date: pd.Timestamp
    observed_date: pd.Timestamp
    value: float


@dataclass(frozen=True)
class SupplyBundle:
    observations: tuple[SupplyObservation, ...]
    manifest: dict


@dataclass(frozen=True)
class _Payload:
    body: bytes
    status_code: int = 200
    attempts: int = 1


def _http_fetch(url: str) -> _Payload:
    """Bounded transport retry; never substitute a source, date, or series."""
    for attempt in range(1, 5):
        try:
            response = requests.get(url, timeout=(10, 45))
        except requests.RequestException as exc:
            if attempt == 4:
                raise StablecoinDataError(
                    f"stablecoin request failed after {attempt} attempts: {url}"
                ) from exc
        else:
            if response.status_code != 429 and not 500 <= response.status_code <= 599:
                return _Payload(response.content, response.status_code, attempt)
            if attempt == 4:
                return _Payload(response.content, response.status_code, attempt)
        time.sleep((0.5, 1.0, 2.0)[attempt - 1])
    raise AssertionError("unreachable stablecoin retry loop")


def _fetch(fetcher: Callable[[str], bytes | _Payload], url: str) -> _Payload:
    try:
        payload = fetcher(url)
    except StablecoinDataError:
        raise
    except Exception as exc:
        raise StablecoinDataError(f"stablecoin request failed: {url}") from exc
    if isinstance(payload, bytes):
        payload = _Payload(payload)
    if not isinstance(payload, _Payload) or not isinstance(payload.body, bytes):
        raise StablecoinDataError(f"unexpected stablecoin response type: {url}")
    if payload.status_code != 200 or not payload.body:
        raise StablecoinDataError(f"stablecoin HTTP {payload.status_code} or empty body: {url}")
    return payload


def _json(body: bytes, label: str) -> object:
    try:
        return json.loads(body.decode("utf-8-sig", errors="strict"))
    except (ValueError, UnicodeError) as exc:
        raise StablecoinDataError(f"{label}: invalid UTF-8 JSON") from exc


def _catalog(body: bytes) -> int:
    parsed = _json(body, "catalog")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("peggedAssets"), list):
        raise StablecoinDataError("catalog: missing peggedAssets list")
    assets = parsed["peggedAssets"]
    seen: dict[str, dict] = {}
    for item in assets:
        if not isinstance(item, dict):
            raise StablecoinDataError("catalog: asset is not an object")
        raw_id = item.get("id")
        if isinstance(raw_id, bool):
            continue
        asset_id = str(raw_id)
        if asset_id in EXPECTED_IDS:
            if asset_id in seen:
                raise StablecoinDataError(f"catalog: duplicate ID {asset_id}")
            seen[asset_id] = item
    for asset_id, symbol in EXPECTED_IDS.items():
        item = seen.get(asset_id)
        if item is None or item.get("symbol") != symbol or item.get("pegType") != "peggedUSD":
            raise StablecoinDataError(f"catalog: ID {asset_id} is not {symbol} peggedUSD")
    return len(assets)


def _date(raw: object, symbol: str) -> pd.Timestamp:
    if not isinstance(raw, str) or not re.fullmatch(r"\d{10}", raw):
        raise StablecoinDataError(f"{symbol}: date must be 10-digit UNIX seconds")
    seconds = int(raw)
    if seconds % 86400:
        raise StablecoinDataError(f"{symbol}: date is not UTC midnight")
    try:
        return pd.Timestamp(datetime.fromtimestamp(seconds, tz=timezone.utc))
    except (OverflowError, OSError, ValueError) as exc:
        raise StablecoinDataError(f"{symbol}: date outside supported range") from exc


def _history(body: bytes, symbol: str) -> tuple[dict[pd.Timestamp, float], dict]:
    parsed = _json(body, symbol)
    if not isinstance(parsed, list):
        raise StablecoinDataError(f"{symbol}: history is not an array")
    days: dict[pd.Timestamp, float] = {}
    all_dates: list[pd.Timestamp] = []
    for row in parsed:
        if not isinstance(row, dict):
            raise StablecoinDataError(f"{symbol}: history row is not an object")
        day = _date(row.get("date"), symbol)
        all_dates.append(day)
        if day < FIRST_DAY or day > LAST_DAY:
            continue
        if day in days:
            raise StablecoinDataError(f"{symbol}: duplicate day {day.date()}")
        circulating = row.get("totalCirculating")
        if not isinstance(circulating, dict) or "peggedUSD" not in circulating:
            raise StablecoinDataError(f"{symbol}: missing totalCirculating.peggedUSD")
        raw_value = circulating["peggedUSD"]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise StablecoinDataError(f"{symbol}: circulating amount must be numeric")
        try:
            value = float(raw_value)
        except (OverflowError, ValueError) as exc:
            raise StablecoinDataError(f"{symbol}: circulating amount out of range") from exc
        if not math.isfinite(value) or value <= 0:
            raise StablecoinDataError(f"{symbol}: circulating amount nonfinite or nonpositive")
        days[day] = value
    grid = pd.date_range(FIRST_DAY, LAST_DAY, freq="D", tz="UTC")
    if len(grid) != EXPECTED_DAYS or len(days) != EXPECTED_DAYS or set(days) != set(grid):
        raise StablecoinDataError(f"{symbol}: 1,891-day UTC history grid incomplete")
    return days, {
        "source_row_count": len(parsed),
        "used_row_count": len(days),
        "source_first_date": min(all_dates).date().isoformat() if all_dates else None,
        "source_last_date": max(all_dates).date().isoformat() if all_dates else None,
        "used_first_date": FIRST_DAY.date().isoformat(),
        "used_last_date": LAST_DAY.date().isoformat(),
        "value_field": "totalCirculating.peggedUSD",
    }


def _response_manifest(url: str, payload: _Payload) -> dict:
    return {
        "url": url,
        "http_status": payload.status_code,
        "http_attempts": payload.attempts,
        "sha256": sha256(payload.body).hexdigest(),
        "bytes": len(payload.body),
    }


def load_and_audit(*, fetcher: Callable[[str], bytes | _Payload] | None = None) -> SupplyBundle:
    """Validate the three fixed responses and form only weekly D+7 observations."""
    fetch = fetcher or _http_fetch
    catalog_payload = _fetch(fetch, CATALOG_URL)
    asset_count = _catalog(catalog_payload.body)
    manifest: dict = {
        "source": BASE_URL,
        "evidence_grade": EVIDENCE_GRADE,
        "independent_oos": False,
        "availability_lag_days": 7,
        "catalog": {**_response_manifest(CATALOG_URL, catalog_payload), "asset_count": asset_count,
                    "verified_ids": dict(EXPECTED_IDS)},
        "histories": {},
    }
    histories: dict[str, dict[pd.Timestamp, float]] = {}
    for symbol, url in HISTORY_URLS.items():
        payload = _fetch(fetch, url)
        days, coverage = _history(payload.body, symbol)
        histories[symbol] = days
        manifest["histories"][symbol] = {**_response_manifest(url, payload), **coverage}
    wednesdays = pd.date_range(FIRST_DAY, LAST_DAY, freq="W-WED", tz="UTC")
    if len(wednesdays) != EXPECTED_WEDNESDAYS:
        raise StablecoinDataError("expected 271 Wednesday observations")
    observations = tuple(
        SupplyObservation(
            release_date=day + AVAILABILITY_LAG,
            observed_date=day,
            value=math.fsum(histories[symbol][day] for symbol in HISTORY_URLS),
        )
        for day in wednesdays
    )
    if any(not math.isfinite(item.value) or item.value <= 0 for item in observations):
        raise StablecoinDataError("combined Wednesday supply nonfinite or nonpositive")
    manifest["weekly_observation_count"] = len(observations)
    manifest["first_observed_date"] = observations[0].observed_date.date().isoformat()
    manifest["last_observed_date"] = observations[-1].observed_date.date().isoformat()
    manifest["first_proxy_release_date"] = observations[0].release_date.date().isoformat()
    manifest["last_proxy_release_date"] = observations[-1].release_date.date().isoformat()
    return SupplyBundle(observations, manifest)
