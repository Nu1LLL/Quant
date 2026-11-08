"""Frozen Deribit option taker-flow source and pinned Binance spot archive."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import re
import threading
import time

import pandas as pd
import requests

from research.btc_exchange_supply_source import SourceFailure, _load_month, _manifest


ENDPOINT = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
FIRST_MONDAY = date(2021, 1, 4)
LAST_MONDAY = date(2025, 12, 29)
OPTION = re.compile(r"BTC-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-([PC])\Z")
_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def _request(start_ms: int, end_ms: int) -> list[dict]:
    params = {"currency": "BTC", "kind": "option", "include_old": "true",
              "count": 1000, "sorting": "asc", "start_timestamp": start_ms,
              "end_timestamp": end_ms}
    last = None
    for attempt, delay in enumerate((0, 1, 2, 4)):
        if delay:
            time.sleep(delay)
        try:
            response = _session().get(ENDPOINT, params=params, timeout=25)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or "error" in payload:
                raise SourceFailure(f"Deribit API error: {payload.get('error') if isinstance(payload, dict) else 'invalid JSON'}")
            result = payload.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("trades"), list) or type(result.get("has_more")) is not bool:
                raise SourceFailure("Deribit trade response schema invalid")
            trades = result["trades"]
            if len(trades) > 1000:
                raise SourceFailure("Deribit trade page exceeded frozen limit")
            if result["has_more"]:
                if start_ms == end_ms:
                    raise SourceFailure("Deribit single millisecond overfull")
                mid = (start_ms + end_ms) // 2
                return _request(start_ms, mid) + _request(mid + 1, end_ms)
            return trades
        except (requests.RequestException, ValueError, SourceFailure) as exc:
            last = exc
            if attempt == 3:
                break
    raise SourceFailure(f"Deribit history request failed after fixed retries: {last}")


def _monday_window(monday: date) -> tuple[int, int]:
    start = datetime(monday.year, monday.month, monday.day, 12, tzinfo=timezone.utc)
    first = int(start.timestamp() * 1000)
    return first, first + 3_600_000 - 1


def _weekly(monday: date) -> tuple[date, int, dict]:
    first, last = _monday_window(monday)
    trades = _request(first, last)
    if not trades:
        raise SourceFailure(f"{monday}: empty Deribit option tape")
    ids: set[tuple[str, str]] = set()
    last_id = -1
    put_signed = put_amount = call_signed = call_amount = 0.0
    put_count = call_count = 0
    canonical = []
    for row in trades:
        if not isinstance(row, dict):
            raise SourceFailure(f"{monday}: non-object trade")
        try:
            trade_id = str(row["trade_id"])
            ident = int(trade_id)
            name = str(row["instrument_name"])
            stamp = int(row["timestamp"])
            key = (trade_id, name)
            if ident < last_id or key in ids or not first <= stamp <= last:
                raise SourceFailure(f"{monday}: trade ID order/uniqueness/time invalid")
            last_id = ident
            ids.add(key)
            canonical.append(row)
            if any(row.get(field) is not None for field in
                   ("block_trade_id", "combo_id", "block_rfq_id", "liquidation")):
                continue
            match = OPTION.fullmatch(name)
            if not match:
                raise SourceFailure(f"{monday}: unexpected option instrument name")
            expiry = datetime.strptime(match[1], "%d%b%y").replace(tzinfo=timezone.utc, hour=8)
            days = (expiry - datetime.fromtimestamp(stamp / 1000, timezone.utc)).total_seconds() / 86400
            strike = float(match[2])
            index = float(row["index_price"])
            amount = float(row["amount"])
            direction = row["direction"]
            if (not all(math.isfinite(x) and x > 0 for x in (strike, index, amount))
                    or direction not in ("buy", "sell")):
                raise SourceFailure(f"{monday}: invalid index/amount/direction")
            ratio = strike / index
            if not 7 <= days <= 60:
                continue
            signed = amount if direction == "buy" else -amount
            if match[3] == "P" and .70 <= ratio < 1:
                put_signed += signed
                put_amount += amount
                put_count += 1
            elif match[3] == "C" and 1 < ratio <= 1.50:
                call_signed += signed
                call_amount += amount
                call_count += 1
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise SourceFailure(f"{monday}: malformed Deribit trade: {exc}") from exc
    low_density = put_count < 5 or call_count < 5
    difference = None if low_density else put_signed / put_amount - call_signed / call_amount
    target = 0 if low_density or difference > 0 else 1
    digest = sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return monday, target, {"week": str(monday), "sha256": digest,
                            "tape_rows": len(trades), "put_rows": put_count,
                            "call_rows": call_count, "low_density": low_density}


def load_sources() -> tuple[dict[date, int], pd.DataFrame, float, dict]:
    mondays = [FIRST_MONDAY + timedelta(days=7 * i) for i in range(261)]
    if mondays[-1] != LAST_MONDAY:
        raise SourceFailure("frozen Deribit Monday calendar invalid")
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
    combined = sha256("\n".join(week_hashes).encode()).hexdigest()
    audit = {"deribit": {"endpoint": ENDPOINT, "weeks": len(weekly),
                         "low_density_weeks": low_density,
                         "coverage_pass": low_density / len(weekly) <= .10,
                         "weekly": [x[2] for x in weekly], "combined_sha256": combined},
             "binance_spot": [x[2] for x in spot_parts]}
    return targets, prices, terminal, audit
