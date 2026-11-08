"""Frozen current-vintage Coin Metrics miner inputs and official BTC spot archives."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from hashlib import sha256
import json
import math

import pandas as pd

from research.btc_exchange_supply_source import (
    SourceFailure as PriceSourceFailure,
    _fetch,
    _load_month,
    _manifest,
)


CM_URL = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?"
          "assets=btc&metrics=IssTotNtv,FeeTotNtv,HashRate,PriceUSD&frequency=1d&"
          "start_time=2020-01-01&end_time=2025-12-31&page_size=10000&paging_from=start")
CM_SHA = "92b245bb2878fdd170e0b884b69564c457c22bb38451f06975f65b079118e3e5"
CM_FIRST = date(2020, 1, 1)
PRICE_FIRST = date(2020, 12, 1)
LAST = date(2025, 12, 31)
METRICS = ("IssTotNtv", "FeeTotNtv", "HashRate", "PriceUSD")


class SourceFailure(ValueError):
    pass


def parse_coinmetrics(body: bytes) -> pd.DataFrame:
    digest = sha256(body).hexdigest()
    if digest != CM_SHA:
        raise SourceFailure(f"Coin Metrics miner-input response changed: {digest}")
    try:
        obj = json.loads(body)
        if not isinstance(obj, dict) or obj.get("next_page_url") is not None:
            raise SourceFailure("Coin Metrics miner-input paging/schema changed")
        rows = obj.get("data")
        expected = (LAST - CM_FIRST).days + 1
        if not isinstance(rows, list) or len(rows) != expected:
            raise SourceFailure("Coin Metrics miner-input day count changed")
        expected_keys = {"asset", "time", *METRICS}
        values = []
        for i, row in enumerate(rows):
            day = CM_FIRST + timedelta(days=i)
            if not isinstance(row, dict) or set(row) != expected_keys or row["asset"] != "btc":
                raise SourceFailure(f"{day}: Coin Metrics miner-input schema changed")
            stamp = pd.Timestamp(row["time"])
            if stamp.tzinfo is None or stamp.tz_convert("UTC") != pd.Timestamp(day, tz="UTC"):
                raise SourceFailure(f"{day}: Coin Metrics miner-input UTC date changed")
            entry = tuple(float(row[key]) for key in METRICS)
            if (not all(math.isfinite(x) for x in entry) or entry[0] < 0 or
                    entry[1] < 0 or entry[2] <= 0 or entry[3] <= 0):
                raise SourceFailure(f"{day}: nonfinite/invalid miner input")
            values.append(entry)
        return pd.DataFrame(values, index=pd.date_range(CM_FIRST, LAST, tz="UTC", freq="D"),
                            columns=list(METRICS))
    except SourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure(f"invalid Coin Metrics miner-input payload: {exc}") from exc


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    try:
        body = _fetch(CM_URL)
        metrics = parse_coinmetrics(body)
        rows = _manifest()
        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(_load_month, rows))
    except PriceSourceFailure as exc:
        raise SourceFailure(str(exc)) from exc
    parts.sort(key=lambda item: item[0])
    terminal = parts[-1][1]
    if not isinstance(terminal, float):
        raise SourceFailure("terminal BTC spot open absent")
    frames = [item[1] for item in parts[:-1]]
    if not all(isinstance(frame, pd.DataFrame) for frame in frames):
        raise SourceFailure("invalid BTC spot archive frame")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(PRICE_FIRST, LAST, tz="UTC", freq="D")):
        raise SourceFailure("BTC spot account date grid invalid")
    audit = {"coinmetrics": {"url": CM_URL, "sha256": CM_SHA,
                             "response_bytes": len(body), "days": len(metrics),
                             "first": str(CM_FIRST), "last": str(LAST),
                             "historical_first_published_vintages_unverified": True},
             "binance_spot": [item[2] for item in parts],
             "reused_price_manifest_commit": "b748b4c"}
    return metrics, prices, terminal, audit
