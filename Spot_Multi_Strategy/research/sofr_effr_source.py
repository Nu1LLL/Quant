"""Frozen SOFR/EFFR source plus official Binance daily spot archives."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from hashlib import sha256
import csv
import io
import math
import subprocess

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure, _manifest, _load_month


RATE_SPECS = {
    "SOFR": {
        "sha": "c408b7a7ec0fe3be2a73fa1bd4d4a2ffa534761fce8afccc89940b2b130febaa",
        "bytes": 35526, "rows": 2243, "first": date(2018, 4, 3),
        "last": date(2026, 11, 5),
        "counts": {2020: 251, 2021: 250, 2022: 249, 2023: 249,
                   2024: 250, 2025: 249},
    },
    "EFFR": {
        "sha": "9738488bf7d195e9032fa8f3cdab10ee07b4115cc58e3ce825f86a95571cce1f",
        "bytes": 108970, "rows": 6874, "first": date(2000, 7, 3),
        "last": date(2026, 11, 5),
        "counts": {2020: 253, 2021: 252, 2022: 250, 2023: 250,
                   2024: 251, 2025: 250},
    },
}


def _fetch_rate(name: str) -> bytes:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={name}"
    proc = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error",
         "--max-time", "25", url], capture_output=True, timeout=30)
    if proc.returncode or not proc.stdout:
        raise SourceFailure(f"{name}: FRED CSV fetch failed: {proc.returncode}")
    return proc.stdout


def parse_rate(name: str, body: bytes) -> dict[date, float | None]:
    spec = RATE_SPECS[name]
    if len(body) != spec["bytes"] or sha256(body).hexdigest() != spec["sha"]:
        raise SourceFailure(f"{name}: FRED current-vintage CSV changed")
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
        if tuple(reader.fieldnames or ()) != ("observation_date", name):
            raise SourceFailure(f"{name}: FRED CSV header changed")
        rows = list(reader)
        if len(rows) != spec["rows"]:
            raise SourceFailure(f"{name}: FRED date-row count changed")
        rates = {}
        previous = None
        for row in rows:
            if set(row) != {"observation_date", name}:
                raise SourceFailure(f"{name}: FRED row shape changed")
            day = date.fromisoformat(row["observation_date"])
            if previous is not None and day <= previous:
                raise SourceFailure(f"{name}: FRED dates duplicate/unsorted")
            previous = day
            raw = row[name]
            if raw == ".":
                value = None
            else:
                value = float(raw)
                if not math.isfinite(value) or abs(value) > 100:
                    raise SourceFailure(f"{name}: nonfinite/out-of-range rate")
            rates[day] = value
        if min(rates) != spec["first"] or max(rates) != spec["last"]:
            raise SourceFailure(f"{name}: FRED date endpoints changed")
        counts = Counter(day.year for day, value in rates.items()
                         if 2020 <= day.year <= 2025 and value is not None)
        if {year: counts[year] for year in spec["counts"]} != spec["counts"]:
            raise SourceFailure(f"{name}: FRED 2020-2025 presence mask changed")
        return rates
    except SourceFailure:
        raise
    except (UnicodeError, csv.Error, ValueError, TypeError, KeyError,
            OverflowError) as exc:
        raise SourceFailure(f"{name}: malformed FRED rate CSV: {exc}") from exc


def load_sources() -> tuple[dict[date, float | None], dict[date, float | None],
                            pd.DataFrame, float, dict]:
    sofr_body, effr_body = _fetch_rate("SOFR"), _fetch_rate("EFFR")
    sofr = parse_rate("SOFR", sofr_body)
    effr = parse_rate("EFFR", effr_body)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, _manifest()))
    parts.sort(key=lambda item: item[0])
    terminal_open = parts[-1][1]
    if not isinstance(terminal_open, float) or not math.isfinite(terminal_open):
        raise SourceFailure("BTC terminal first open absent")
    frames = [part[1] for part in parts[:-1]]
    if not all(isinstance(item, pd.DataFrame) for item in frames):
        raise SourceFailure("BTC spot archive did not contain daily frames")
    prices = pd.concat(frames)
    expected = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
    if not prices.index.equals(expected):
        raise SourceFailure("joined BTC spot daily grid changed")
    audit = {"sofr": {"sha256": RATE_SPECS["SOFR"]["sha"], "bytes": len(sofr_body),
                      "rows": len(sofr), "current_vintage": True},
             "effr": {"sha256": RATE_SPECS["EFFR"]["sha"], "bytes": len(effr_body),
                      "rows": len(effr), "current_vintage": True},
             "binance_spot": [part[2] for part in parts]}
    return sofr, effr, prices, terminal_open, audit
