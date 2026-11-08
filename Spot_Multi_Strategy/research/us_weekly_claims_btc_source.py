"""Frozen current-vintage U.S. weekly claims and audited Binance BTC spot source."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from hashlib import sha256
import csv
import io
import math
import re

import pandas as pd

from research.btc_exchange_supply_source import (
    MANIFEST, SourceFailure, _fetch, _load_month, _manifest,
)


ICSA_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=ICSA"
ICSA_SHA = "f84a9c9c6831eeb41b5b15125008f5d48d72c91453d75ad9e05118d40702f428"
SPOT_MANIFEST_SHA = "581c681e6d14065242ae8ff464b313fca02038fc16a6e158f37a203ae3cd4a5f"
FIRST_WEEK = date(1967, 1, 7)
LAST_WEEK = date(2026, 10, 31)
WEEK_COUNT = 3122
PRICE_FIRST = date(2020, 12, 1)
PRICE_LAST = date(2025, 12, 31)


def parse_icsa(body: bytes) -> dict[date, int | None]:
    if sha256(body).hexdigest() != ICSA_SHA:
        raise SourceFailure("current-vintage FRED ICSA response SHA changed")
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
        if tuple(reader.fieldnames or ()) != ("observation_date", "ICSA"):
            raise SourceFailure("FRED ICSA schema changed")
        rows = list(reader)
        if len(rows) != WEEK_COUNT:
            raise SourceFailure("FRED ICSA weekly row count changed")
        out: dict[date, int | None] = {}
        for index, row in enumerate(rows):
            week = FIRST_WEEK + timedelta(days=7 * index)
            if (set(row) != {"observation_date", "ICSA"}
                    or row["observation_date"] != week.isoformat()):
                raise SourceFailure(f"FRED ICSA date grid changed at {week}")
            value = row["ICSA"]
            if value == ".":
                out[week] = None
            elif re.fullmatch(r"[0-9]+", value or ""):
                number = int(value)
                if number <= 0:
                    raise SourceFailure(f"FRED ICSA nonpositive value at {week}")
                out[week] = number
            else:
                raise SourceFailure(f"FRED ICSA numeric schema changed at {week}")
        if week != LAST_WEEK:
            raise SourceFailure("FRED ICSA final week changed")
        years = {year: sum(w.year == year for w in out) for year in range(2019, 2026)}
        if years != {2019: 52, 2020: 52, 2021: 52, 2022: 53,
                     2023: 52, 2024: 52, 2025: 52}:
            raise SourceFailure("FRED ICSA year coverage changed")
        if any(v is None for w, v in out.items() if 2019 <= w.year <= 2025):
            raise SourceFailure("FRED ICSA missing in 2019–2025")
        return out
    except SourceFailure:
        raise
    except (UnicodeError, csv.Error, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure(f"FRED ICSA CSV invalid: {exc}") from exc


def load_sources() -> tuple[dict[date, int | None], pd.DataFrame, float, dict]:
    try:
        manifest_sha = sha256(MANIFEST.read_bytes()).hexdigest()
    except OSError as exc:
        raise SourceFailure(f"frozen Binance spot manifest missing: {exc}") from exc
    if manifest_sha != SPOT_MANIFEST_SHA:
        raise SourceFailure("frozen Binance spot manifest SHA changed")
    claims_body = _fetch(ICSA_URL)
    claims = parse_icsa(claims_body)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, _manifest()))
    parts.sort(key=lambda item: item[0])
    terminal_open = parts[-1][1]
    if not isinstance(terminal_open, float) or not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("terminal BTC spot open absent")
    frames = [part[1] for part in parts[:-1]]
    if not all(isinstance(frame, pd.DataFrame) for frame in frames):
        raise SourceFailure("nonterminal BTC spot frame invalid")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(PRICE_FIRST, PRICE_LAST, tz="UTC", freq="D")):
        raise SourceFailure("joined BTC spot date grid changed")
    return claims, prices, terminal_open, {
        "fred_icsa": {"url": ICSA_URL, "sha256": sha256(claims_body).hexdigest(),
                      "bytes": len(claims_body), "weeks": len(claims),
                      "current_vintage_not_point_in_time": True},
        "binance_spot": [part[2] for part in parts],
    }
