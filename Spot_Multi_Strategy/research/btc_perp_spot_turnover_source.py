"""Pinned Binance BTC spot and USD-M perpetual daily turnover archives."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import csv
import math
import re
import zipfile

import pandas as pd

from research.btc_exchange_supply_source import _fetch, _load_month, _manifest, SourceFailure
from research.coinbase_premium_source import _parse_bn_zip, SourceFailure as SpotParseFailure


ROOT = Path(__file__).resolve().parent
FUTURES_MANIFEST = ROOT / "preregistrations/2026-11-08_btc_futures_spot_turnover_futures_manifest.csv"
SPOT_WARMUP_MANIFEST = ROOT / "preregistrations/2026-11-08_cme_option_btc_warmup_manifest.csv"
FIRST = date(2020, 11, 1)
LAST = date(2025, 12, 31)
HEADER = ("open_time", "open", "high", "low", "close", "volume", "close_time",
          "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore")


def _futures_records() -> list[dict[str, str]]:
    with FUTURES_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = pd.period_range("2020-11", "2025-12", freq="M").astype(str).tolist()
    if [x.get("month") for x in rows] != expected:
        raise SourceFailure("BTC futures manifest month grid changed")
    for item in rows:
        month = item["month"]
        url = ("https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/"
               f"BTCUSDT-1d-{month}.zip")
        if item.get("url") != url or not re.fullmatch(r"[0-9a-f]{64}", item.get("official_zip_sha256", "")):
            raise SourceFailure(f"{month}: BTC futures manifest URL/hash invalid")
    return rows


def _parse_futures_zip(body: bytes, month: str) -> dict[date, float]:
    name = f"BTCUSDT-1d-{month}.csv"
    try:
        with zipfile.ZipFile(BytesIO(body)) as archive:
            if archive.namelist() != [name]:
                raise SourceFailure(f"{month}: BTC futures ZIP member changed")
            rows = list(csv.reader(archive.read(name).decode("utf-8-sig").splitlines()))
        if rows and rows[0][0] == "open_time":
            if tuple(rows[0]) != HEADER:
                raise SourceFailure(f"{month}: BTC futures header changed")
            rows = rows[1:]
        first = date.fromisoformat(month + "-01")
        next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        if len(rows) != (next_month - first).days:
            raise SourceFailure(f"{month}: BTC futures daily row count changed")
        values: dict[date, float] = {}
        for i, row in enumerate(rows):
            day = first + timedelta(days=i)
            expected = int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp()) * 1000
            if (len(row) != 12 or not re.fullmatch(r"[0-9]{13}", row[0])
                    or int(row[0]) != expected or int(row[6]) != expected + 86_400_000 - 1):
                raise SourceFailure(f"{month}: BTC futures timestamp/schema invalid at {day}")
            op, hi, lo, cl, volume = [float(row[j]) for j in (1, 2, 3, 4, 7)]
            if (not all(math.isfinite(x) for x in (op, hi, lo, cl, volume))
                    or lo <= 0 or hi < max(op, cl) or lo > min(op, cl) or volume <= 0):
                raise SourceFailure(f"{month}: BTC futures OHLC/quote volume invalid at {day}")
            values[day] = volume
        return values
    except SourceFailure:
        raise
    except (zipfile.BadZipFile, RuntimeError, UnicodeError, csv.Error,
            ValueError, TypeError, IndexError, OverflowError) as exc:
        raise SourceFailure(f"{month}: BTC futures CSV invalid: {exc}") from exc


def _futures_month(item: dict[str, str]) -> tuple[str, dict[date, float], dict]:
    month, url, pin = item["month"], item["url"], item["official_zip_sha256"]
    checksum = _fetch(url + ".CHECKSUM").decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-f]{64})\s+" + re.escape(url.rsplit("/", 1)[-1]), checksum)
    if match is None or match.group(1) != pin:
        raise SourceFailure(f"{month}: official BTC futures checksum changed")
    body = _fetch(url)
    digest = sha256(body).hexdigest()
    if digest != pin:
        raise SourceFailure(f"{month}: BTC futures ZIP SHA changed")
    values = _parse_futures_zip(body, month)
    return month, values, {"month": month, "url": url, "sha256": digest,
                           "bytes": len(body), "days": len(values)}


def _spot_november() -> tuple[str, pd.DataFrame, dict]:
    with SPOT_WARMUP_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    item = next((x for x in rows if x.get("month") == "2020-11"), None)
    url = ("https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/"
           "BTCUSDT-1d-2020-11.zip")
    if item is None or item.get("url") != url or not re.fullmatch(r"[0-9a-f]{64}", item.get("official_zip_sha256", "")):
        raise SourceFailure("pinned November 2020 spot warmup absent")
    pin = item["official_zip_sha256"]
    checksum = _fetch(url + ".CHECKSUM").decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-f]{64})\s+BTCUSDT-1d-2020-11\.zip", checksum)
    if match is None or match.group(1) != pin:
        raise SourceFailure("November 2020 spot official checksum changed")
    body = _fetch(url)
    digest = sha256(body).hexdigest()
    if digest != pin:
        raise SourceFailure("November 2020 spot ZIP SHA changed")
    try:
        frame, audit = _parse_bn_zip(body, "2020-11", False)
    except SpotParseFailure as exc:
        raise SourceFailure(str(exc)) from exc
    if not isinstance(frame, pd.DataFrame):
        raise SourceFailure("November 2020 spot frame absent")
    audit.update({"url": url, "sha256": digest, "bytes": len(body)})
    return "2020-11", frame, audit


def load_sources() -> tuple[dict[date, float], pd.DataFrame, float, dict]:
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_parts = list(pool.map(_futures_month, _futures_records()))
    future_parts.sort(key=lambda x: x[0])
    futures: dict[date, float] = {}
    for _, values, _ in future_parts:
        for day, value in values.items():
            if day in futures:
                raise SourceFailure(f"duplicate futures day {day}")
            futures[day] = value
    expected = pd.date_range(FIRST, LAST, tz="UTC", freq="D")
    if list(futures) != [x.date() for x in expected]:
        raise SourceFailure("joined BTC futures date grid changed")
    with ThreadPoolExecutor(max_workers=4) as pool:
        spot_parts = list(pool.map(_load_month, _manifest()))
    spot_parts.append(_spot_november())
    spot_parts.sort(key=lambda x: x[0])
    terminal = spot_parts[-1][1]
    if not isinstance(terminal, float) or terminal <= 0:
        raise SourceFailure("terminal BTC spot open absent")
    frames = [x[1] for x in spot_parts[:-1]]
    if not all(isinstance(x, pd.DataFrame) for x in frames):
        raise SourceFailure("nonterminal BTC spot frame invalid")
    prices = pd.concat(frames)
    if not prices.index.equals(expected):
        raise SourceFailure("joined BTC spot date grid changed")
    audit = {"futures": [x[2] for x in future_parts],
             "spot": [x[2] for x in spot_parts]}
    return futures, prices, terminal, audit
