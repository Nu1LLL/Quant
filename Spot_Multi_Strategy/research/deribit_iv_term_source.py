"""Frozen, read-only source for the preregistered BTC IV maturity-slope test."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import math
import re
from statistics import median

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure, _load_month, _manifest
from research.crypto_blackbox.deribit_history_archive import ArchiveWindow, fetch_window


FIRST_MONDAY = date(2021, 1, 4)
LAST_MONDAY = date(2025, 12, 29)
OPTION = re.compile(r"BTC-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-([PC])\Z")
EXCLUSION_FIELDS = ("block_trade_id", "combo_id", "block_rfq_id", "liquidation")


def _monday_window(monday: date) -> tuple[int, int]:
    start = datetime(monday.year, monday.month, monday.day, 12, tzinfo=timezone.utc)
    first = int(start.timestamp() * 1000)
    return first, first + 3_599_999


def _validated_bucket(row: dict) -> tuple[str | None, float]:
    """Validate every option before eligibility; return its bucket and IV."""
    if not isinstance(row, dict):
        raise SourceFailure("non-object Deribit option trade")
    try:
        name = row["instrument_name"]
        match = OPTION.fullmatch(name) if isinstance(name, str) else None
        if match is None:
            raise SourceFailure("malformed BTC option instrument")
        expiry = datetime.strptime(match[1], "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
        stamp = row["timestamp"]
        if type(stamp) is not int:
            raise SourceFailure("non-integer Deribit trade timestamp")
        strike = float(match[2])
        index = float(row["index_price"])
        amount = float(row["amount"])
        iv = float(row["iv"])
        if not (all(math.isfinite(x) and x > 0 for x in (strike, index, amount, iv))
                and iv <= 1000):
            raise SourceFailure("invalid option strike/index/amount/IV")
        if any(row.get(key) is not None for key in EXCLUSION_FIELDS):
            return None, iv
        ratio = strike / index
        days = (expiry - datetime.fromtimestamp(stamp / 1000, timezone.utc)).total_seconds() / 86400
        if not .90 <= ratio <= 1.10:
            return None, iv
        if 7 <= days <= 21:
            return "front", iv
        if 45 <= days <= 90:
            return "back", iv
        return None, iv
    except SourceFailure:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SourceFailure(f"malformed Deribit option trade: {type(exc).__name__}") from exc


def _week_from_archive(monday: date, archive: ArchiveWindow) -> tuple[date, int, dict]:
    first, last = _monday_window(monday)
    if (archive.currency != "BTC" or archive.kind != "option"
            or archive.start_ms != first or archive.end_ms != last or not archive.rows):
        raise SourceFailure(f"{monday}: invalid or empty Deribit archive window")
    buckets: dict[str, list[float]] = {"front": [], "back": []}
    for row in archive.rows:
        bucket, iv = _validated_bucket(row)
        if bucket is not None:
            buckets[bucket].append(iv)
    low_density = min(map(len, buckets.values())) < 5
    target = 0 if low_density else int(median(buckets["front"]) <= median(buckets["back"]))
    return monday, target, {"week": str(monday),
                            "sha256": archive.canonical_sha256,
                            "tape_rows": len(archive.rows),
                            "front_rows": len(buckets["front"]),
                            "back_rows": len(buckets["back"]),
                            "low_density": low_density,
                            "archive_requests": archive.requests,
                            "split_windows": archive.split_windows}


def _weekly(monday: date) -> tuple[date, int, dict]:
    first, last = _monday_window(monday)
    try:
        return _week_from_archive(monday, fetch_window("BTC", "option", first, last))
    except (ValueError, OSError) as exc:
        raise SourceFailure(f"{monday}: Deribit archive failed: {type(exc).__name__}: {exc}") from exc


def load_sources() -> tuple[dict[date, int], pd.DataFrame, float, dict]:
    mondays = [FIRST_MONDAY + timedelta(days=7 * i) for i in range(261)]
    if mondays[-1] != LAST_MONDAY:
        raise SourceFailure("frozen Monday calendar invalid")
    with ThreadPoolExecutor(max_workers=4) as pool:
        weekly = list(pool.map(_weekly, mondays))
    low_density = sum(item[2]["low_density"] for item in weekly)
    targets = {day: target for day, target, _ in weekly}
    with ThreadPoolExecutor(max_workers=4) as pool:
        spot_parts = list(pool.map(_load_month, _manifest()))
    spot_parts.sort(key=lambda item: item[0])
    if [x[0] for x in spot_parts] != pd.period_range("2020-12", "2026-01", freq="M").astype(str).tolist():
        raise SourceFailure("pinned BTC spot month grid invalid")
    terminal = spot_parts[-1][1]
    if not isinstance(terminal, float) or not math.isfinite(terminal) or terminal <= 0:
        raise SourceFailure("BTC terminal opening invalid")
    prices = pd.concat([x[1] for x in spot_parts[:-1]])
    expected = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
    if not prices.index.equals(expected):
        raise SourceFailure("joined BTC spot daily grid invalid")
    week_hashes = [x[2]["sha256"] for x in weekly]
    audit = {"deribit": {"weeks": len(weekly), "low_density_weeks": low_density,
                         "coverage_pass": low_density / len(weekly) <= .15,
                         "combined_sha256": sha256("\n".join(week_hashes).encode()).hexdigest(),
                         "weekly": [x[2] for x in weekly]},
             "binance_spot": [x[2] for x in spot_parts]}
    return targets, prices, terminal, audit
