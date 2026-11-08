"""Pinned current-vintage mempool observations and checksum-audited BTC bars."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure, _fetch, _load_month, _manifest


URL = ("https://api.blockchain.info/charts/mempool-size?timespan=3287days&"
       "start=2017-01-01&sampled=false&rollingAverage=1minute&format=json")
PIN = "d7ed28e3a60ffb1ecabfa3e60461cae45792349ae9548e50be477894e13a9038"
MANIFEST_SHA = "581c681e6d14065242ae8ff464b313fca02038fc16a6e158f37a203ae3cd4a5f"
FIRST_X = 1_483_229_700  # 2017-01-01 00:15 UTC
LAST_X = 1_767_225_600   # 2026-01-01 00:00 UTC
EXPECTED_ROWS = 315_432


def parse_mempool(body: bytes) -> tuple[dict[date, float], dict[date, int]]:
    if len(body) != 10_131_759 or sha256(body).hexdigest() != PIN:
        raise SourceFailure("Blockchain.com historical mempool response changed")
    try:
        obj = json.loads(body)
        if (not isinstance(obj, dict)
                or set(obj) != {"status", "name", "unit", "period", "description", "values"}
                or obj["status"] != "ok" or obj["name"] != "Mempool Size"
                or obj["unit"] != "Bytes" or obj["period"] != "minute"
                or not isinstance(obj["description"], str)):
            raise SourceFailure("Blockchain.com mempool schema/unit changed")
        rows = obj["values"]
        if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
            raise SourceFailure("Blockchain.com mempool row count changed")
        by_day: dict[date, list[float]] = defaultdict(list)
        previous: int | None = None
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"x", "y"}:
                raise SourceFailure("Blockchain.com mempool row schema changed")
            timestamp, raw = row["x"], row["y"]
            if (isinstance(timestamp, bool) or not isinstance(timestamp, int)
                    or timestamp % 900 or (previous is not None and timestamp <= previous)
                    or (index == 0 and timestamp != FIRST_X)
                    or (index == len(rows) - 1 and timestamp != LAST_X)):
                raise SourceFailure("Blockchain.com mempool UTC grid changed")
            previous = timestamp
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise SourceFailure("Blockchain.com mempool value not numeric")
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise SourceFailure("Blockchain.com mempool value invalid")
            day = datetime.fromtimestamp(timestamp, timezone.utc).date()
            if day <= date(2025, 12, 31):
                by_day[day].append(value)
        medians = {day: float(median(values)) for day, values in by_day.items()}
        counts = {day: len(values) for day, values in by_day.items()}
        return medians, counts
    except SourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure(f"Blockchain.com mempool response invalid: {exc}") from exc


def load_sources() -> tuple[dict[date, float], dict[date, int], pd.DataFrame, float, dict]:
    body = _fetch(URL)
    medians, counts = parse_mempool(body)
    manifest = Path(__file__).parent / "preregistrations/2026-11-08_btc_exchange_supply_spot_manifest.csv"
    if sha256(manifest.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise SourceFailure("frozen Binance spot manifest changed")
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, _manifest()))
    parts.sort(key=lambda item: item[0])
    terminal_open = parts[-1][1]
    if not isinstance(terminal_open, float) or not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("terminal BTC spot open invalid")
    frames = [part[1] for part in parts[:-1]]
    if not all(isinstance(frame, pd.DataFrame) for frame in frames):
        raise SourceFailure("nonterminal BTC spot month invalid")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(date(2020, 12, 1), date(2025, 12, 31), tz="UTC", freq="D")):
        raise SourceFailure("BTC spot daily grid invalid")
    audit = {"mempool": {"url": URL, "sha256": PIN, "response_bytes": len(body),
                         "rows": EXPECTED_ROWS, "daily_medians": len(medians),
                         "current_vintage_not_point_in_time": True},
             "binance_spot": [part[2] for part in parts]}
    return medians, counts, prices, terminal_open, audit
