"""Frozen FRED WALCL current vintage and official Binance BTC spot archives."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from hashlib import sha256
from io import StringIO
import csv
import math
import re

import pandas as pd

from research.btc_exchange_supply_source import _fetch, _load_month, _manifest, SourceFailure


WALCL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WALCL&cosd=2020-01-01&coed=2025-12-31"
WALCL_SHA = "bb37a64b0bd604647a1db85a0bfa6247da27b17bad3b8e42de222e5d178e11a4"
FIRST = date(2020, 1, 1)
LAST = date(2025, 12, 31)


def parse_walcl(body: bytes) -> dict[date, float]:
    digest = sha256(body).hexdigest()
    if digest != WALCL_SHA:
        raise SourceFailure(f"frozen FRED WALCL response changed: {digest}")
    try:
        rows = list(csv.reader(StringIO(body.decode("utf-8-sig")), strict=True))
        if not rows or rows[0] != ["observation_date", "WALCL"] or len(rows) != 315:
            raise SourceFailure("FRED WALCL CSV header or row count changed")
        output: dict[date, float] = {}
        for i, row in enumerate(rows[1:]):
            expected = FIRST + timedelta(days=7 * i)
            if (len(row) != 2 or row[0] != expected.isoformat() or
                    not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", row[1])):
                raise SourceFailure(f"FRED WALCL grid or numeric schema changed at row {i}")
            value = float(row[1])
            if not math.isfinite(value) or value <= 0:
                raise SourceFailure(f"FRED WALCL value invalid at {expected}")
            output[expected] = value
        if len(output) != 314 or next(reversed(output)) != LAST:
            raise SourceFailure("FRED WALCL dates not complete")
        return output
    except SourceFailure:
        raise
    except (csv.Error, ValueError, UnicodeError, TypeError, OverflowError) as exc:
        raise SourceFailure(f"invalid frozen FRED WALCL CSV: {exc}") from exc


def load_sources() -> tuple[dict[date, float], pd.DataFrame, float, dict]:
    body = _fetch(WALCL_URL)
    walcl = parse_walcl(body)
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
    audit = {"fred_walcl": {"url": WALCL_URL, "raw_sha256": WALCL_SHA,
                             "rows": 314, "unit": "million_usd",
                             "current_vintage_not_each_week_first_release": True},
             "binance_spot": [x[2] for x in parts]}
    return walcl, prices, terminal, audit
