"""Frozen U.S. Treasury daily TGA balance source for one crypto study.

This module does not fetch the official numeric source until the experiment
runner is invoked after preregistration and implementation have been pushed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time as midnight, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from math import isfinite
from typing import Callable
from urllib.parse import urlparse

import requests


BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance"
PARAMS = {
    "fields": "record_date,account_type,close_today_bal,open_today_bal",
    "filter": "record_date:gte:2020-11-01,record_date:lte:2026-09-30",
    "page[size]": "10000",
}
FIRST = date(2020, 11, 1)
LAST = date(2026, 9, 30)
LAYOUT_CHANGE = date(2022, 4, 18)
YEAR_COUNTS = {2020: 41, 2021: 252, 2022: 250, 2023: 250,
               2024: 251, 2025: 250, 2026: 189}
MAX_BODY_BYTES = 2 * 1024 * 1024


class TgaSourceError(RuntimeError):
    """Official source is unavailable or violates the frozen data contract."""


@dataclass(frozen=True)
class TgaObservation:
    observed_date: date
    balance_million_usd: float
    proxy_available_at: datetime


@dataclass(frozen=True)
class TgaBundle:
    observations: tuple[TgaObservation, ...]
    manifest: dict


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TgaSourceError("duplicate FiscalData JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise TgaSourceError(f"nonfinite FiscalData JSON value: {value}")


def _download() -> tuple[bytes, str]:
    for attempt in range(3):
        try:
            with requests.get(BASE, params=PARAMS, timeout=(10, 45), stream=True) as response:
                if urlparse(response.url).hostname != "api.fiscaldata.treasury.gov":
                    raise TgaSourceError("FiscalData redirected off official host")
                if response.status_code == 200:
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=65_536):
                        body.extend(chunk)
                        if len(body) > MAX_BODY_BYTES:
                            raise TgaSourceError("FiscalData response exceeds two megabytes")
                    if body:
                        return bytes(body), response.url
                elif response.status_code not in (429, 500, 502, 503, 504):
                    raise TgaSourceError(f"FiscalData HTTP {response.status_code}")
                if attempt == 2:
                    raise TgaSourceError(f"FiscalData HTTP {response.status_code} or empty body")
        except requests.RequestException as exc:
            if attempt == 2:
                raise TgaSourceError("FiscalData request failed") from exc
        import time
        time.sleep(0.5 * 2**attempt)
    raise AssertionError("unreachable")


def _balance_for_row(row: dict, observed: date) -> float | None:
    label = row["account_type"]
    if observed < LAYOUT_CHANGE:
        if label not in ("Federal Reserve Account", "Treasury General Account (TGA)"):
            return None
        raw = row["close_today_bal"]
    else:
        if label != "Treasury General Account (TGA) Closing Balance":
            return None
        raw = row["open_today_bal"]
    if not isinstance(raw, str) or raw.strip() != raw:
        raise TgaSourceError(f"missing or malformed TGA balance at {observed}")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise TgaSourceError(f"nonnumeric TGA balance at {observed}") from exc
    if not value.is_finite() or not 0 < value < 5_000_000:
        raise TgaSourceError(f"out-of-range TGA balance at {observed}")
    result = float(value)
    if not isfinite(result):
        raise TgaSourceError(f"nonfinite TGA balance at {observed}")
    return result


def _parse(body: bytes, final_url: str) -> TgaBundle:
    if not isinstance(body, bytes) or not body or len(body) > MAX_BODY_BYTES:
        raise TgaSourceError("empty or oversized FiscalData response")
    if urlparse(final_url).hostname != "api.fiscaldata.treasury.gov":
        raise TgaSourceError("untrusted FiscalData final URL")
    try:
        doc = json.loads(body.decode("utf-8"), object_pairs_hook=_unique,
                         parse_constant=_reject_constant)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise TgaSourceError("invalid FiscalData JSON") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("meta"), dict):
        raise TgaSourceError("FiscalData metadata missing")
    meta = doc["meta"]
    rows = doc.get("data")
    if (not isinstance(rows, list) or len(rows) != 5566
            or type(meta.get("total-count")) is not int or meta["total-count"] != 5566
            or type(meta.get("count")) is not int or meta["count"] != 5566
            or type(meta.get("total-pages")) is not int or meta["total-pages"] != 1):
        raise TgaSourceError("FiscalData pagination or registered row count changed")
    values: dict[date, float] = {}
    labels: Counter[str] = Counter()
    required = {"record_date", "account_type", "close_today_bal", "open_today_bal"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise TgaSourceError("FiscalData row schema changed")
        try:
            observed = date.fromisoformat(row["record_date"])
        except (TypeError, ValueError) as exc:
            raise TgaSourceError("invalid FiscalData record date") from exc
        if not FIRST <= observed <= LAST or not isinstance(row["account_type"], str):
            raise TgaSourceError("FiscalData row outside registered source")
        balance = _balance_for_row(row, observed)
        if balance is None:
            continue
        if observed in values:
            raise TgaSourceError(f"duplicate TGA daily balance at {observed}")
        values[observed] = balance
        labels[row["account_type"]] += 1
    dates = sorted(values)
    counts = Counter(day.year for day in dates)
    if (len(dates) != 1483 or dict(counts) != YEAR_COUNTS or
            any((b - a).days > 4 for a, b in zip(dates, dates[1:]))):
        raise TgaSourceError("registered TGA daily coverage changed")
    observations = tuple(TgaObservation(
        day, values[day], datetime.combine(day + timedelta(days=4), midnight.min,
                                            timezone.utc)) for day in dates)
    manifest = {"url": final_url, "http_status": 200, "bytes": len(body),
                "sha256": sha256(body).hexdigest(), "source_rows": len(rows),
                "accepted_days": len(dates), "year_counts": dict(counts),
                "account_labels": dict(labels),
                "first_date": dates[0].isoformat(), "last_date": dates[-1].isoformat(),
                "unit": "million_usd", "proxy_lag_calendar_days": 4}
    return TgaBundle(observations, manifest)


def load_and_audit(*, fetcher: Callable[[], tuple[bytes, str]] | None = None) -> TgaBundle:
    body, final_url = (fetcher or _download)()
    return _parse(body, final_url)
