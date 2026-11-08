"""Pinned Treasury DTS TGA balances and checksum-audited BTC spot bars."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from hashlib import sha256
import json
import math
import re

import pandas as pd
import requests

from research.btc_exchange_supply_source import SourceFailure, _fetch, _load_month, _manifest


TGA_BASE = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
            "v1/accounting/dts/operating_cash_balance")
TGA_PARAMS = (("fields", "record_date,account_type,open_today_bal,close_today_bal"),
              ("filter", "record_date:gte:2021-01-01,record_date:lte:2025-12-31"),
              ("sort", "record_date,account_type"), ("page[size]", "10000"))
TGA_URL = requests.Request("GET", TGA_BASE, params=TGA_PARAMS).prepare().url
TGA_SHA = "05a02d5584029e00b6bf61aa0a9cae117a2ea5a2be15cd351a8fadbb0c4f31ba"
TGA_FIELDS = {"record_date", "account_type", "open_today_bal", "close_today_bal"}
TARGET_LABELS = {"Federal Reserve Account", "Treasury General Account (TGA)",
                 "Treasury General Account (TGA) Closing Balance"}
START = date(2021, 1, 1)
LAST = date(2025, 12, 31)


def _target(day: date) -> tuple[str, str]:
    if day <= date(2021, 9, 30):
        return "Federal Reserve Account", "close_today_bal"
    if day <= date(2022, 4, 15):
        return "Treasury General Account (TGA)", "close_today_bal"
    return "Treasury General Account (TGA) Closing Balance", "open_today_bal"


def parse_tga(body: bytes) -> dict[date, float]:
    digest = sha256(body).hexdigest()
    if digest != TGA_SHA:
        raise SourceFailure(f"Treasury DTS historic response changed: {digest}")
    try:
        obj = json.loads(body)
        if not isinstance(obj, dict) or set(obj) != {"data", "meta", "links"}:
            raise SourceFailure("Treasury DTS response envelope changed")
        meta, rows = obj["meta"], obj["data"]
        if (not isinstance(meta, dict) or not isinstance(rows, list) or len(rows) != 4687
                or meta.get("count") != 4687 or meta.get("total-count") != 4687
                or meta.get("total-pages") != 1):
            raise SourceFailure("Treasury DTS pagination/row count changed")
        if (meta.get("dataFormats", {}).get("close_today_bal") != "$1,000,000"
                or meta.get("dataFormats", {}).get("open_today_bal") != "$1,000,000"):
            raise SourceFailure("Treasury DTS USD-millions unit changed")
        all_days: set[date] = set()
        values: dict[date, float] = {}
        prior_key: tuple[date, str] | None = None
        label_counts = {label: 0 for label in TARGET_LABELS}
        for row in rows:
            if not isinstance(row, dict) or set(row) != TGA_FIELDS:
                raise SourceFailure("Treasury DTS row schema changed")
            day = date.fromisoformat(row["record_date"])
            label = row["account_type"]
            if not START <= day <= LAST or not isinstance(label, str):
                raise SourceFailure("Treasury DTS date/label invalid")
            key = (day, label)
            if prior_key is not None and key <= prior_key:
                raise SourceFailure("Treasury DTS rows duplicate/unsorted")
            prior_key = key
            all_days.add(day)
            if label in TARGET_LABELS:
                wanted, field = _target(day)
                if label != wanted or day in values:
                    raise SourceFailure(f"Treasury DTS TGA phase/duplicate invalid at {day}")
                raw = row[field]
                if not isinstance(raw, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw):
                    raise SourceFailure(f"Treasury DTS TGA numeric schema invalid at {day}")
                number = float(raw)
                if not math.isfinite(number) or number <= 0:
                    raise SourceFailure(f"Treasury DTS TGA balance invalid at {day}")
                values[day] = number
                label_counts[label] += 1
        if (len(all_days) != 1253 or len(values) != 1253 or set(values) != all_days
                or min(values) != date(2021, 1, 4) or max(values) != LAST
                or label_counts != {"Federal Reserve Account": 189,
                                    "Treasury General Account (TGA)": 136,
                                    "Treasury General Account (TGA) Closing Balance": 928}):
            raise SourceFailure("Treasury DTS TGA daily coverage/phase counts changed")
        return values
    except SourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure(f"Treasury DTS response invalid: {exc}") from exc


def load_sources() -> tuple[dict[date, float], pd.DataFrame, float, dict]:
    tga_body = _fetch(TGA_URL)
    tga = parse_tga(tga_body)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, _manifest()))
    parts.sort(key=lambda x: x[0])
    terminal_open = parts[-1][1]
    if not isinstance(terminal_open, float) or not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("terminal BTC spot open absent")
    frames = [x[1] for x in parts[:-1]]
    if not all(isinstance(x, pd.DataFrame) for x in frames):
        raise SourceFailure("nonterminal BTC spot frame invalid")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(date(2020, 12, 1), LAST, tz="UTC", freq="D")):
        raise SourceFailure("joined BTC spot date grid changed")
    audit = {"treasury_tga": {"url": TGA_URL, "sha256": sha256(tga_body).hexdigest(),
                              "bytes": len(tga_body), "all_rows": 4687,
                              "target_days": len(tga),
                              "current_vintage_not_point_in_time": True},
             "binance_spot": [x[2] for x in parts]}
    return tga, prices, terminal_open, audit
