"""Pinned DeFiLlama parent-protocol daily revenue, without signal construction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import time

import pandas as pd
import requests


FIRST = date(2020, 12, 7)
LAST = date(2025, 12, 31)
PARENTS = (
    ("AAVEUSDT", "aave", "parent#aave", "AAVE",
     "8ee4584b186a876fdbf7a50c7fda2b322559f948d5003a9f86ac28d014fa08f7"),
    ("COMPUSDT", "compound-finance", "parent#compound-finance", "COMP",
     "7b68bdc3e12cb2dd168bcbe1acd5fc85100aa9c144e9b40ffb18aacdf0dc2bab"),
    ("CRVUSDT", "curve-finance", "parent#curve-finance", "CRV",
     "08e5a6fc2bd8040c0c88b083da833315874cf6c588106b125a2a159048ab28bb"),
)


class RevenueSourceFailure(ValueError):
    pass


def _fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, headers={"User-Agent": "MiniMedallionResearch/1.0"},
                                    timeout=35)
            response.raise_for_status()
            if len(response.content) > 5_000_000:
                raise RevenueSourceFailure("oversized DeFiLlama summary")
            return response.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(attempt + 1)
    raise RevenueSourceFailure(f"DeFiLlama request failed: {url}: {last}")


def parse_parent(body: bytes, parent: tuple[str, str, str, str, str]
                 ) -> tuple[list[float], dict]:
    symbol, slug, parent_id, ticker, pin = parent
    try:
        obj = json.loads(body)
        if not isinstance(obj, dict) or obj.get("defillamaId") != parent_id or obj.get("symbol") != ticker:
            raise RevenueSourceFailure(f"{slug}: protocol identity changed")
        chart = obj.get("totalDataChart")
        if not isinstance(chart, list):
            raise RevenueSourceFailure(f"{slug}: historical revenue chart absent")
        selected = []
        for row in chart:
            if not isinstance(row, list) or len(row) != 2 or type(row[0]) is not int:
                raise RevenueSourceFailure(f"{slug}: malformed chart row")
            day = datetime.fromtimestamp(row[0], timezone.utc).date()
            if FIRST <= day <= LAST:
                selected.append(row)
        expected = (LAST - FIRST).days + 1
        if len(selected) != expected:
            raise RevenueSourceFailure(f"{slug}: incomplete historical revenue grid")
        canonical = json.dumps(selected, separators=(",", ":"), ensure_ascii=False,
                               allow_nan=False).encode()
        digest = sha256(canonical).hexdigest()
        if digest != pin:
            raise RevenueSourceFailure(f"{slug}: historical revenue subset changed: {digest}")
        values = []
        for i, row in enumerate(selected):
            day = FIRST + timedelta(days=i)
            if row[0] != int(datetime.combine(day, datetime.min.time(),
                                              timezone.utc).timestamp()):
                raise RevenueSourceFailure(f"{slug}: missing, duplicate or non-midnight day {day}")
            value = row[1]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise RevenueSourceFailure(f"{slug}: nonfinite or negative daily revenue {day}")
            values.append(float(value))
        return values, {"symbol": symbol, "slug": slug, "protocol_id": parent_id,
                        "ticker": ticker, "url": f"https://api.llama.fi/summary/fees/{slug}?dataType=dailyRevenue",
                        "subset_sha256": digest, "subset_days": len(selected),
                        "first": str(FIRST), "last": str(LAST),
                        "response_bytes": len(body),
                        "historical_first_published_vintage_unavailable": True}
    except RevenueSourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise RevenueSourceFailure(f"{slug}: malformed DeFiLlama response: {exc}") from exc


def _load_one(parent: tuple[str, str, str, str, str]) -> tuple[list[float], dict]:
    slug = parent[1]
    body = _fetch(f"https://api.llama.fi/summary/fees/{slug}?dataType=dailyRevenue")
    return parse_parent(body, parent)


def load_revenue() -> tuple[pd.DataFrame, list[dict]]:
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(_load_one, PARENTS))
    frame = pd.DataFrame({parent[0]: result[0] for parent, result in zip(PARENTS, results)},
                         index=pd.date_range(FIRST, LAST, tz="UTC", freq="D"))
    return frame, [result[1] for result in results]
