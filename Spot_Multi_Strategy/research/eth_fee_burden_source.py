"""Pinned ETH block-space fee burden and Binance spot source; no signals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re

import pandas as pd

from research.eth_defi_tvl_source import SourceFailure, _fetch, _load_month


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "preregistrations/2026-11-08_eth_defi_tvl_spot_manifest.csv"
METRIC_URL = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?"
              "assets=eth&metrics=FeeTotNtv,TxCnt&frequency=1d&"
              "start_time=2020-11-01&end_time=2025-12-31&"
              "page_size=10000&paging_from=start")
METRIC_SHA = "639886b3eec9f3f3825366c1c0e00b9b1ff7dce48d68e1670cf16e6e6b6ffc7d"
FIRST = date(2020, 11, 1)
LAST = date(2025, 12, 31)
COUNT = (LAST - FIRST).days + 1
_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")
_INTEGER = re.compile(r"[0-9]+\Z")


def parse_metrics(body: bytes) -> tuple[dict[date, float], dict[date, int]]:
    digest = sha256(body).hexdigest()
    if digest != METRIC_SHA:
        raise SourceFailure(f"ETH fee/transaction response changed: {digest}")
    try:
        doc = json.loads(body)
        if (not isinstance(doc, dict) or set(doc) - {"data", "next_page_url"}
                or doc.get("next_page_url") is not None):
            raise SourceFailure("ETH fee/transaction response pagination/schema changed")
        rows = doc.get("data")
        if not isinstance(rows, list) or len(rows) != COUNT:
            raise SourceFailure("ETH fee/transaction UTC row count changed")
        fees: dict[date, float] = {}
        transactions: dict[date, int] = {}
        for i, row in enumerate(rows):
            day = FIRST + timedelta(days=i)
            if (not isinstance(row, dict)
                    or set(row) != {"asset", "time", "FeeTotNtv", "TxCnt"}
                    or row["asset"] != "eth"
                    or row["time"] != day.isoformat() + "T00:00:00.000000000Z"):
                raise SourceFailure(f"ETH fee/transaction identity/grid failed at {day}")
            fee_raw, tx_raw = row["FeeTotNtv"], row["TxCnt"]
            if (not isinstance(fee_raw, str) or not _DECIMAL.fullmatch(fee_raw)
                    or not isinstance(tx_raw, str) or not _INTEGER.fullmatch(tx_raw)):
                raise SourceFailure(f"ETH fee/transaction numeric schema failed at {day}")
            fee = Decimal(fee_raw)
            tx = int(tx_raw)
            if not fee.is_finite() or fee < 0 or tx <= 0:
                raise SourceFailure(f"ETH fee/transaction nonpositive/nonfinite at {day}")
            numeric = float(fee)
            if not 0 <= numeric < float("inf"):
                raise SourceFailure(f"ETH fee/transaction overflow at {day}")
            fees[day], transactions[day] = numeric, tx
        return fees, transactions
    except SourceFailure:
        raise
    except (TypeError, ValueError, KeyError, OverflowError, InvalidOperation) as exc:
        raise SourceFailure(f"ETH fee/transaction response malformed: {exc}") from exc


def load_sources() -> tuple[dict[date, float], dict[date, int], pd.DataFrame, float, dict]:
    body = _fetch(METRIC_URL)
    if len(body) > 1_000_000:
        raise SourceFailure("ETH fee/transaction response oversized")
    fees, transactions = parse_metrics(body)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if tuple(reader.fieldnames or ()) != ("month", "url", "official_zip_sha256"):
            raise SourceFailure("ETH spot manifest schema changed")
        records = list(reader)
    expected = pd.period_range("2020-11", "2026-01", freq="M").astype(str).tolist()
    if [row.get("month") for row in records] != expected:
        raise SourceFailure("ETH spot manifest month grid changed")
    for row in records:
        month = row["month"]
        url = ("https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1d/"
               f"ETHUSDT-1d-{month}.zip")
        if row.get("url") != url or not re.fullmatch(
                r"[0-9a-f]{64}", row.get("official_zip_sha256", "")):
            raise SourceFailure(f"ETH spot manifest URL/hash changed at {month}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, records))
    parts.sort(key=lambda item: item[0])
    terminal = parts[-1][1]
    frames = [item[1] for item in parts[:-1]]
    if not isinstance(terminal, float) or not all(isinstance(x, pd.DataFrame) for x in frames):
        raise SourceFailure("ETH spot terminal/prior archive type failed")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(FIRST, LAST, tz="UTC", freq="D")):
        raise SourceFailure("ETH spot joined UTC grid failed")
    audit = {"coinmetrics": {"url": METRIC_URL, "sha256": METRIC_SHA,
                              "rows": COUNT, "first": str(FIRST), "last": str(LAST),
                              "current_vintage_not_point_in_time": True},
             "binance_spot": [part[2] for part in parts]}
    return fees, transactions, prices, terminal, audit
