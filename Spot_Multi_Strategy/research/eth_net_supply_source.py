"""Frozen ETH ledger supply and checksum-audited Binance spot archive source."""

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
import zipfile

import pandas as pd

from research.eth_defi_tvl_source import _fetch, _parse_spot_zip, SourceFailure


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "preregistrations/2026-11-08_eth_net_supply_spot_manifest.csv"
SUPPLY_URL = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?"
              "assets=eth&metrics=SplyCur&frequency=1d&start_time=2021-07-01&"
              "end_time=2026-09-30&page_size=10000&paging_from=start")
SUPPLY_SHA = "edc9f4f54e908246efe739f42dc2011ee176e6c7269f45d50252018c13157714"
SUPPLY_FIRST = date(2021, 7, 1)
PRICE_FIRST = date(2021, 8, 1)
PRICE_LAST = date(2026, 9, 30)
TERMINAL = date(2026, 10, 1)
COUNT = 1918


def parse_supply(body: bytes) -> dict[date, float]:
    digest = sha256(body).hexdigest()
    if digest != SUPPLY_SHA:
        raise SourceFailure(f"ETH supply response changed: {digest}")
    try:
        doc = json.loads(body)
        if not isinstance(doc, dict) or doc.get("next_page_url") is not None:
            raise SourceFailure("ETH supply response pagination changed")
        rows = doc.get("data")
        if not isinstance(rows, list) or len(rows) != COUNT:
            raise SourceFailure("ETH supply response count changed")
        values: dict[date, float] = {}
        for i, row in enumerate(rows):
            expected = SUPPLY_FIRST + timedelta(days=i)
            if (not isinstance(row, dict) or set(row) != {"asset", "time", "SplyCur"}
                    or row["asset"] != "eth" or
                    row["time"] != expected.isoformat() + "T00:00:00.000000000Z"):
                raise SourceFailure(f"ETH supply identity/date/schema changed at {expected}")
            raw = row["SplyCur"]
            if not isinstance(raw, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw):
                raise SourceFailure(f"ETH supply numeric schema changed at {expected}")
            value = float(raw)
            if not math.isfinite(value) or value <= 0:
                raise SourceFailure(f"ETH supply invalid at {expected}")
            values[expected] = value
        if len(values) != COUNT or next(reversed(values)) != PRICE_LAST:
            raise SourceFailure("ETH supply day grid invalid")
        return values
    except SourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure(f"invalid ETH supply response: {exc}") from exc


def _records() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = pd.period_range("2021-08", "2026-10", freq="M").astype(str).tolist()
    if [x.get("month") for x in rows] != expected:
        raise SourceFailure("ETH spot manifest month grid invalid")
    for record in rows:
        month = record["month"]
        url = ("https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1d/"
               f"ETHUSDT-1d-{month}.zip")
        if record.get("url") != url or not re.fullmatch(r"[0-9a-f]{64}", record.get("official_zip_sha256", "")):
            raise SourceFailure(f"ETH spot manifest URL/hash invalid at {month}")
    return rows


def _terminal_open(body: bytes) -> float:
    try:
        with zipfile.ZipFile(BytesIO(body)) as archive:
            if archive.namelist() != ["ETHUSDT-1d-2026-10.csv"]:
                raise SourceFailure("terminal ETH ZIP member invalid")
            with archive.open("ETHUSDT-1d-2026-10.csv") as member:
                first_line = member.readline()
        row = next(csv.reader([first_line.decode("utf-8-sig")]))
        expected = int(datetime.combine(TERMINAL, datetime.min.time(), timezone.utc).timestamp()) * 1_000_000
        value = float(row[1])
        if len(row) != 12 or int(row[0]) != expected or not math.isfinite(value) or value <= 0:
            raise SourceFailure("terminal ETH opening timestamp/price invalid")
        return value
    except SourceFailure:
        raise
    except (zipfile.BadZipFile, RuntimeError, UnicodeError, ValueError, IndexError) as exc:
        raise SourceFailure(f"terminal ETH ZIP invalid: {exc}") from exc


def _month(record: dict[str, str]) -> tuple[str, pd.DataFrame | float, dict]:
    month, url, pin = record["month"], record["url"], record["official_zip_sha256"]
    checksum = _fetch(url + ".CHECKSUM").decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-f]{64})\s+" + re.escape(url.rsplit("/", 1)[-1]), checksum)
    if match is None or match.group(1) != pin:
        raise SourceFailure(f"{month}: official ETH spot checksum changed")
    body = _fetch(url)
    digest = sha256(body).hexdigest()
    if digest != pin:
        raise SourceFailure(f"{month}: ETH spot ZIP SHA changed")
    if month == "2026-10":
        data: pd.DataFrame | float = _terminal_open(body)
        detail = {"month": month, "terminal_first_open_only": True}
    else:
        data, detail = _parse_spot_zip(body, month, False)
    detail.update({"url": url, "sha256": digest, "bytes": len(body)})
    return month, data, detail


def load_sources() -> tuple[dict[date, float], pd.DataFrame, float, dict]:
    body = _fetch(SUPPLY_URL)
    supply = parse_supply(body)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_month, _records()))
    parts.sort(key=lambda x: x[0])
    terminal = parts[-1][1]
    if not isinstance(terminal, float) or terminal <= 0:
        raise SourceFailure("terminal ETH opening price absent")
    frames = [x[1] for x in parts[:-1]]
    if not all(isinstance(x, pd.DataFrame) for x in frames):
        raise SourceFailure("nonterminal ETH spot frame invalid")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(PRICE_FIRST, PRICE_LAST, tz="UTC", freq="D")):
        raise SourceFailure("joined ETH spot UTC grid invalid")
    audit = {"coinmetrics_supply": {"url": SUPPLY_URL, "sha256": SUPPLY_SHA,
                                    "rows": COUNT, "first": str(SUPPLY_FIRST),
                                    "last": str(PRICE_LAST), "unit": "native_eth",
                                    "current_vintage_not_first_published": True},
             "binance_spot": [x[2] for x in parts]}
    return supply, prices, terminal, audit
