"""Frozen BSC/all-chain stablecoin share and official BNB spot sources."""

from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import json
import math
import re
from zipfile import BadZipFile, ZipFile

import pandas as pd

from research.stablecoin_liquidity_source import (
    LiquiditySourceFailure as SourceFailure,
    _fetch, _load_month, _manifest,
)


FIRST = date(2020, 12, 1)
LAST = date(2025, 12, 31)
TERMINAL = date(2026, 1, 1)
COUNT = (LAST - FIRST).days + 1
CHARTS = (
    ("BSC", "be5d580e1fb1e36a11dcf5d8a9a4de04ad37ec5617fc8c4466949f273212a5a0"),
    ("all", "28f1afbed6a5c29f9bd053b3fe55cc7ca9dc6a72596ebeeca95646a3cf020301"),
)


def parse_chart(body: bytes, chain: str, pin: str) -> dict[date, float]:
    try:
        rows = json.loads(body)
        if not isinstance(rows, list):
            raise SourceFailure(f"{chain}: stablecoin chart is not a list")
        start = int(datetime.combine(FIRST, datetime.min.time(), timezone.utc).timestamp())
        end = int(datetime.combine(LAST, datetime.min.time(), timezone.utc).timestamp())
        subset: list[list[int]] = []
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("date", "")).isdecimal():
                continue
            stamp = int(row["date"])
            if start <= stamp <= end:
                if type(row["date"]) is not str:
                    raise SourceFailure(f"{chain}: selected UTC date is not a decimal string")
                total = row.get("totalCirculating")
                subset.append([stamp, total.get("peggedUSD") if isinstance(total, dict) else None])
        canonical = json.dumps(subset, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode()
        digest = sha256(canonical).hexdigest()
        if digest != pin:
            raise SourceFailure(f"{chain}: stablecoin historical subset changed: {digest}")
        if len(subset) != COUNT:
            raise SourceFailure(f"{chain}: stablecoin daily count changed")
        values: dict[date, float] = {}
        for i, (stamp, raw) in enumerate(subset):
            day = FIRST + timedelta(days=i)
            if stamp != start + i * 86_400 or type(raw) is not int or raw <= 0:
                raise SourceFailure(f"{chain}: invalid UTC/value row {day}")
            value = float(raw)
            if not math.isfinite(value) or value <= 0:
                raise SourceFailure(f"{chain}: nonfinite stablecoin amount at {day}")
            values[day] = value
        return values
    except SourceFailure:
        raise
    except (TypeError, ValueError, OverflowError, KeyError) as exc:
        raise SourceFailure(f"{chain}: invalid stablecoin chart: {exc}") from exc


def _chart(spec: tuple[str, str]) -> tuple[str, dict[date, float], dict]:
    chain, pin = spec
    url = f"https://stablecoins.llama.fi/stablecoincharts/{chain}"
    body = _fetch(url, limit=2_000_000)
    amounts = parse_chart(body, chain, pin)
    return chain, amounts, {"url": url, "selected_subset_sha256": pin,
                            "rows": COUNT, "first": str(FIRST), "last": str(LAST),
                            "response_bytes": len(body),
                            "current_vintage_not_first_published": True}


def _terminal_grid(record: dict[str, str]) -> int:
    """Check all January timestamps but read only the terminal first open."""
    url = ("https://data.binance.vision/data/spot/monthly/klines/BNBUSDT/1d/"
           "BNBUSDT-1d-2026-01.zip")
    if record["month"] != "2026-01":
        raise SourceFailure("wrong terminal BNB archive")
    body = _fetch(url, limit=100_000)
    if sha256(body).hexdigest() != record["sha256"]:
        raise SourceFailure("BNB terminal archive changed during UTC-grid audit")
    start = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
    try:
        with ZipFile(BytesIO(body)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != "BNBUSDT-1d-2026-01.csv":
                raise SourceFailure("BNB terminal ZIP member changed")
            with archive.open(members[0]) as handle:
                count = 0
                for row in csv.reader(TextIOWrapper(handle, encoding="utf-8-sig", newline=""),
                                      strict=True):
                    stamp = start + count * 86_400 * 1_000_000
                    if (len(row) != 12 or row[0] != str(stamp)
                            or row[6] != str(stamp + 86_400 * 1_000_000 - 1)):
                        raise SourceFailure("BNB terminal UTC grid changed")
                    count += 1
    except SourceFailure:
        raise
    except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
        raise SourceFailure(f"BNB terminal ZIP unreadable: {exc}") from exc
    if count != monthrange(2026, 1)[1]:
        raise SourceFailure("BNB terminal January day count changed")
    return count


def load_sources() -> tuple[dict[date, float], pd.DataFrame, float, dict]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        charts = list(pool.map(_chart, CHARTS))
    chain, global_ = charts[0][1], charts[1][1]
    share: dict[date, float] = {}
    for day in (FIRST + timedelta(days=i) for i in range(COUNT)):
        ratio = chain[day] / global_[day]
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise SourceFailure(f"BSC share invalid on {day}")
        share[day] = ratio
    records = [row for row in _manifest() if row["symbol"] == "BNBUSDT"]
    if len(records) != 62:
        raise SourceFailure("BNB spot manifest count changed")
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, records))
    parts.sort(key=lambda part: part[0])
    terminal = parts[-1][2]
    frames = [part[2] for part in parts[:-1]]
    if not isinstance(terminal, float) or not all(isinstance(x, pd.DataFrame) for x in frames):
        raise SourceFailure("BNB spot terminal/history type invalid")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(FIRST, LAST, tz="UTC", freq="D")):
        raise SourceFailure("BNB spot joined UTC grid invalid")
    terminal_count = _terminal_grid(records[-1])
    audit = {"stablecoin_charts": {item[0]: item[2] for item in charts},
             "spot": [part[3] for part in parts],
             "terminal_grid_days": terminal_count}
    return share, prices, terminal, audit
