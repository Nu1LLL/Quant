"""Pinned Coin Metrics Community native issuance and current supply reader."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import time

import pandas as pd
import requests


FIRST = date(2020, 12, 1)
LAST = date(2025, 12, 31)
ASSETS = (
    ("BCHUSDT", "bch", "fc55c664bbd229554210af4eb0c43e2ef5f55fd79b3a6d514af1bc76405d358b"),
    ("DOGEUSDT", "doge", "57e3a66d5c6767dd6610f83a4d2b8db8da2821153b7cf10d3c5b0b9dbe118990"),
    ("ETCUSDT", "etc", "df339631f713e7a6d8252524114544ff7cc0e60d25333318c2490472284dc437"),
)
URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FIELDS = {"asset", "time", "IssTotNtv", "SplyCur"}


class IssuanceSourceFailure(ValueError):
    pass


def _fetch(asset: str) -> bytes:
    params = {"assets": asset, "metrics": "IssTotNtv,SplyCur", "frequency": "1d",
              "start_time": str(FIRST), "end_time": str(LAST), "page_size": 10000,
              "paging_from": "start"}
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(URL, params=params,
                                    headers={"User-Agent": "MiniMedallionResearch/1.0"},
                                    timeout=35)
            response.raise_for_status()
            if len(response.content) > 2_000_000:
                raise IssuanceSourceFailure("oversized Coin Metrics response")
            return response.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(attempt + 1)
    raise IssuanceSourceFailure(f"Coin Metrics request failed for {asset}: {last}")


def parse_asset(body: bytes, asset: tuple[str, str, str]
                ) -> tuple[dict[tuple[str, str], list[float]], dict]:
    symbol, network_id, pin = asset
    try:
        response = json.loads(body)
        if not isinstance(response, dict) or set(response) != {"data"}:
            raise IssuanceSourceFailure(f"{network_id}: response schema changed or paginated")
        rows = response["data"]
        if not isinstance(rows, list) or len(rows) != (LAST - FIRST).days + 1:
            raise IssuanceSourceFailure(f"{network_id}: incomplete daily rows")
        canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode()
        digest = sha256(canonical).hexdigest()
        if digest != pin:
            raise IssuanceSourceFailure(f"{network_id}: source history changed: {digest}")
        series = {(symbol, "IssTotNtv"): [], (symbol, "SplyCur"): []}
        for i, row in enumerate(rows):
            day = FIRST + timedelta(days=i)
            stamp = datetime.combine(day, datetime.min.time(), timezone.utc).strftime(
                "%Y-%m-%dT00:00:00.000000000Z")
            if not isinstance(row, dict) or set(row) != FIELDS or row["asset"] != network_id or row["time"] != stamp:
                raise IssuanceSourceFailure(f"{network_id}: missing/duplicate UTC day or field {day}")
            for key in ("IssTotNtv", "SplyCur"):
                value = float(row[key])
                if not math.isfinite(value) or value <= 0:
                    raise IssuanceSourceFailure(f"{network_id}: invalid positive {key} on {day}")
                series[(symbol, key)].append(value)
        return series, {"symbol": symbol, "network_id": network_id,
                        "canonical_data_sha256": digest, "daily_rows": len(rows),
                        "first": str(FIRST), "last": str(LAST),
                        "response_bytes": len(body),
                        "historical_first_published_vintage_unavailable": True}
    except IssuanceSourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise IssuanceSourceFailure(f"{network_id}: malformed Coin Metrics data: {exc}") from exc


def _load_one(asset: tuple[str, str, str]):
    return parse_asset(_fetch(asset[1]), asset)


def load_issuance() -> tuple[pd.DataFrame, list[dict]]:
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(_load_one, ASSETS))
    columns = {}
    for series, _ in results:
        columns.update(series)
    frame = pd.DataFrame(columns, index=pd.date_range(FIRST, LAST, tz="UTC", freq="D"))
    return frame, [audit for _, audit in results]
