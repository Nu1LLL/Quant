"""Strict frozen source audit for BTC exchange-inventory drain, no signal logic."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import csv
import json
import math
import re
import time
import zipfile

import pandas as pd
import requests

from research.coinbase_premium_source import SourceFailure as PriceParseFailure
from research.coinbase_premium_source import _parse_bn_zip


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "preregistrations/2026-11-08_btc_exchange_supply_spot_manifest.csv"
CM_URL = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?"
          "assets=btc&metrics=SplyExNtv&frequency=1d&start_time=2020-10-01&"
          "end_time=2025-12-31&page_size=10000&paging_from=start")
CM_SHA = "8b5bebe377060eb89a168fbe5633c06d28b1550aa6669579a773e03968d1bdcc"
METRIC = "SplyExNtv"
CM_FIRST = date(2020, 10, 1)
PRICE_FIRST = date(2020, 12, 1)
LAST = date(2025, 12, 31)
TERMINAL = date(2026, 1, 1)


class SourceFailure(ValueError):
    pass


def _fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, headers={
                "User-Agent": "MiniMedallionResearch/1.0 (noncommercial source audit)"},
                timeout=35)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(attempt + 1)
    raise SourceFailure(f"source request failed: {url}: {last}")


def parse_coinmetrics(body: bytes) -> dict[date, float]:
    digest = sha256(body).hexdigest()
    if digest != CM_SHA:
        raise SourceFailure(f"Coin Metrics historic response changed: {digest}")
    try:
        obj = json.loads(body)
        if not isinstance(obj, dict) or obj.get("next_page_url") is not None:
            raise SourceFailure("Coin Metrics response paging/schema changed")
        rows = obj.get("data")
        expected = (LAST - CM_FIRST).days + 1
        if not isinstance(rows, list) or len(rows) != expected:
            raise SourceFailure("Coin Metrics metric day count changed")
        parsed: dict[date, float] = {}
        expected_keys = {"asset", "time", METRIC, METRIC + "-status", METRIC + "-status-time"}
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != expected_keys or row["asset"] != "btc":
                raise SourceFailure("Coin Metrics asset/metric schema changed")
            stamp = pd.Timestamp(row["time"])
            day = CM_FIRST + timedelta(days=i)
            if stamp.tzinfo is None or stamp.tz_convert("UTC") != pd.Timestamp(day, tz="UTC"):
                raise SourceFailure(f"Coin Metrics UTC date grid changed at {day}")
            value = float(row[METRIC])
            if not math.isfinite(value) or value <= 0 or day in parsed:
                raise SourceFailure(f"Coin Metrics invalid native supply at {day}")
            parsed[day] = value
        return parsed
    except SourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure(f"invalid Coin Metrics payload: {exc}") from exc


def _manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    months = pd.period_range("2020-12", "2026-01", freq="M").astype(str).tolist()
    if [x["month"] for x in rows] != months:
        raise SourceFailure("Binance spot manifest month grid changed")
    for item in rows:
        url = ("https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/"
               f"BTCUSDT-1d-{item['month']}.zip")
        if item["url"] != url or len(item["official_zip_sha256"]) != 64:
            raise SourceFailure("Binance spot manifest URL/hash changed")
    return rows


def _terminal_open(body: bytes) -> float:
    try:
        with zipfile.ZipFile(BytesIO(body)) as archive:
            if archive.namelist() != ["BTCUSDT-1d-2026-01.csv"]:
                raise SourceFailure("terminal Binance ZIP member invalid")
            first_line = archive.read("BTCUSDT-1d-2026-01.csv").splitlines()[0]
        row = next(csv.reader([first_line.decode("utf-8-sig")]))
        timestamp = int(row[0])
        expected = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1000000
        op = float(row[1])
        if len(row) != 12 or timestamp != expected or not math.isfinite(op) or op <= 0:
            raise SourceFailure("terminal January first open invalid")
        return op
    except SourceFailure:
        raise
    except (zipfile.BadZipFile, ValueError, IndexError, UnicodeError, RuntimeError) as exc:
        raise SourceFailure(f"invalid terminal Binance ZIP: {exc}") from exc


def _load_month(item: dict[str, str]) -> tuple[str, pd.DataFrame | float, dict]:
    month, url, pin = item["month"], item["url"], item["official_zip_sha256"]
    checksum = _fetch(url + ".CHECKSUM").decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-f]{64})\s+" + re.escape(url.rsplit("/", 1)[-1]), checksum)
    if match is None or match.group(1) != pin:
        raise SourceFailure(f"{month}: official Binance checksum changed")
    body = _fetch(url)
    digest = sha256(body).hexdigest()
    if digest != pin:
        raise SourceFailure(f"{month}: Binance ZIP does not match frozen SHA")
    if month == "2026-01":
        data: pd.DataFrame | float = _terminal_open(body)
        detail = {"month": month, "terminal_first_open_only": True}
    else:
        try:
            data, detail = _parse_bn_zip(body, month, False)
        except PriceParseFailure as exc:
            raise SourceFailure(str(exc)) from exc
    detail.update({"url": url, "sha256": digest, "bytes": len(body)})
    return month, data, detail


def load_sources() -> tuple[dict[date, float], pd.DataFrame, float, dict]:
    supply_body = _fetch(CM_URL)
    supply = parse_coinmetrics(supply_body)
    rows = _manifest()
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, rows))
    parts.sort(key=lambda x: x[0])
    terminal = parts[-1][1]
    if not isinstance(terminal, float):
        raise SourceFailure("terminal spot open absent")
    frames = [x[1] for x in parts[:-1]]
    if not all(isinstance(x, pd.DataFrame) for x in frames):
        raise SourceFailure("nonterminal spot archive malformed")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(PRICE_FIRST, LAST, tz="UTC", freq="D")):
        raise SourceFailure("joined Binance BTC spot grid invalid")
    audit = {"coinmetrics": {"url": CM_URL, "sha256": CM_SHA,
                             "response_bytes": len(supply_body), "days": len(supply),
                             "first": str(CM_FIRST), "last": str(LAST),
                             "current_wallet_label_vintage_not_point_in_time": True},
             "binance_spot": [x[2] for x in parts]}
    return supply, prices, terminal, audit
