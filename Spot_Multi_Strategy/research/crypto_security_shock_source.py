"""Frozen current-vintage hack catalog and previously pinned BTC spot bars."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math

import pandas as pd

from research.btc_exchange_supply_source import _fetch, _load_month, _manifest, SourceFailure


HACK_URL = "https://api.llama.fi/hacks"
HACK_SHA = "9fa5a45e806bf8718fab0c7e007f35311ed6280c92e69dae4e38344ebd812f8f"
FIRST = date(2020, 11, 1)
LAST = date(2025, 12, 31)


def parse_hacks(body: bytes) -> dict[date, int]:
    try:
        data = json.loads(body)
        if not isinstance(data, list):
            raise SourceFailure("hack catalog is not a list")
        rows = []
        for item in data:
            if (not isinstance(item, dict) or type(item.get("date")) is not int
                    or "name" not in item or "amount" not in item):
                raise SourceFailure("hack catalog date/schema invalid")
            stamp = datetime.fromtimestamp(item["date"], timezone.utc)
            if FIRST <= stamp.date() <= LAST:
                if stamp.time() != datetime.min.time() or not isinstance(item.get("name"), str) or not item["name"]:
                    raise SourceFailure("hack date/name invalid")
                amount = item.get("amount")
                if amount is not None and (type(amount) is not int or amount < 0):
                    raise SourceFailure("hack dollar amount invalid")
                rows.append({"date": item["date"], "name": item["name"], "amount": amount})
        rows.sort(key=lambda x: (x["date"], str(x["name"]), str(x["amount"])))
        canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode()
        digest = sha256(canonical).hexdigest()
        if digest != HACK_SHA or len(rows) != 892:
            raise SourceFailure(f"frozen hack catalog changed: {digest}; rows={len(rows)}")
        if sum(x["amount"] is None for x in rows) != 27:
            raise SourceFailure("frozen null amount count changed")
        if (datetime.fromtimestamp(rows[0]["date"], timezone.utc).date() != date(2020, 11, 2)
                or datetime.fromtimestamp(rows[-1]["date"], timezone.utc).date() != date(2025, 12, 30)):
            raise SourceFailure("frozen hack date bounds changed")
        totals: dict[date, int] = {}
        for row in rows:
            day = datetime.fromtimestamp(row["date"], timezone.utc).date()
            totals[day] = totals.get(day, 0) + (row["amount"] or 0)
        return totals
    except SourceFailure:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SourceFailure(f"invalid hack catalog: {exc}") from exc


def load_sources() -> tuple[dict[date, int], pd.DataFrame, float, dict]:
    body = _fetch(HACK_URL)
    events = parse_hacks(body)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, _manifest()))
    parts.sort(key=lambda x: x[0])
    terminal = parts[-1][1]
    if not isinstance(terminal, float) or not math.isfinite(terminal) or terminal <= 0:
        raise SourceFailure("terminal spot open invalid")
    frames = [x[1] for x in parts[:-1]]
    if not all(isinstance(x, pd.DataFrame) for x in frames):
        raise SourceFailure("spot month frame invalid")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(date(2020, 12, 1), LAST, tz="UTC", freq="D")):
        raise SourceFailure("joined BTC spot UTC grid invalid")
    audit = {"hacks": {"url": HACK_URL, "canonical_sha256": HACK_SHA,
                        "projected_rows": 892, "null_amount_rows": 27,
                        "current_vintage_without_first_publication_timestamps": True},
             "binance_spot": [x[2] for x in parts]}
    return events, prices, terminal, audit
