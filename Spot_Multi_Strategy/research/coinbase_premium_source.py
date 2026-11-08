"""Official-source audit for one frozen corrected-Coinbase-premium experiment."""

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


ROOT = Path(__file__).resolve().parent
CB_MANIFEST = ROOT / "preregistrations/2026-11-08_coinbase_premium_candles_manifest.csv"
BN_MANIFEST = ROOT / "preregistrations/2026-11-08_coinbase_premium_binance_spot_manifest.csv"
CB_START = date(2021, 9, 1)
BN_START = date(2021, 8, 1)
LAST = date(2026, 9, 30)
TERMINAL = date(2026, 10, 1)
USER_AGENT = "MiniMedallionSourceAudit/1.0"


class SourceFailure(ValueError):
    pass


def _fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=35)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(attempt + 1)
    raise SourceFailure(f"source request failed: {url}: {last}")


def _coinbase_manifest() -> list[dict[str, str]]:
    with CB_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 16 or set(x["pair"] for x in rows) != {"BTC-USD", "USDT-USD"}:
        raise SourceFailure("Coinbase manifest product/partition count changed")
    for pair in ("BTC-USD", "USDT-USD"):
        pieces = [r for r in rows if r["pair"] == pair]
        expected = CB_START
        for item in pieces:
            start, end = date.fromisoformat(item["first_date"]), date.fromisoformat(item["last_date"])
            if start != expected or end < start or (end - start).days + 1 != int(item["days"]):
                raise SourceFailure(f"Coinbase {pair} manifest date grid changed")
            url = (f"https://api.exchange.coinbase.com/products/{pair}/candles?granularity=86400"
                   f"&start={start.isoformat()}T00%3A00%3A00Z&end={end.isoformat()}T00%3A00%3A00Z")
            if item["url"] != url or len(item["response_sha256"]) != 64:
                raise SourceFailure(f"Coinbase {pair} manifest URL/hash changed")
            expected = end + timedelta(days=1)
        if expected != LAST + timedelta(days=1):
            raise SourceFailure(f"Coinbase {pair} manifest terminal date changed")
    return rows


def _load_cb_part(item: dict[str, str]) -> tuple[str, pd.DataFrame, dict]:
    body = _fetch(item["url"])
    digest = sha256(body).hexdigest()
    if digest != item["response_sha256"] or len(body) != int(item["bytes"]):
        raise SourceFailure(f"Coinbase {item['pair']} {item['first_date']}: response revision")
    try:
        raw = json.loads(body)
        if not isinstance(raw, list) or len(raw) != int(item["days"]):
            raise SourceFailure("Coinbase candle count/schema invalid")
        parsed = []
        for row in raw:
            if not isinstance(row, list) or len(row) != 6:
                raise SourceFailure("Coinbase candle shape invalid")
            stamp = int(row[0])
            if stamp != row[0] or stamp % 86400:
                raise SourceFailure("Coinbase candle time invalid")
            lo, hi, op, cl, vol = [float(v) for v in row[1:]]
            if not all(math.isfinite(v) for v in (lo, hi, op, cl, vol)):
                raise SourceFailure("Coinbase nonfinite candle")
            if lo <= 0 or hi < max(op, cl) or lo > min(op, cl) or vol <= 0:
                raise SourceFailure("Coinbase invalid OHLC/volume")
            parsed.append((datetime.fromtimestamp(stamp, timezone.utc), lo, hi, op, cl, vol))
        parsed.sort(key=lambda x: x[0])
        dates = pd.DatetimeIndex([x[0] for x in parsed])
        first, last = date.fromisoformat(item["first_date"]), date.fromisoformat(item["last_date"])
        expected = pd.date_range(first, last, tz="UTC", freq="D")
        if not dates.equals(expected):
            raise SourceFailure("Coinbase missing/duplicate/extra day")
        frame = pd.DataFrame([x[1:] for x in parsed], index=dates,
                             columns=["low", "high", "open", "close", "volume"])
        audit = {"pair": item["pair"], "first": str(first), "last": str(last),
                 "url": item["url"], "sha256": digest, "bytes": len(body), "days": len(frame)}
        return item["pair"], frame, audit
    except SourceFailure:
        raise
    except (ValueError, TypeError, IndexError, OverflowError) as exc:
        raise SourceFailure(f"Coinbase {item['pair']} {item['first_date']}: {exc}") from exc


def _binance_manifest() -> list[dict[str, str]]:
    with BN_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    months = pd.period_range("2021-08", "2026-10", freq="M").astype(str).tolist()
    if [r["month"] for r in rows] != months:
        raise SourceFailure("Binance manifest month grid changed")
    for item in rows:
        month = item["month"]
        url = ("https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/"
               f"BTCUSDT-1d-{month}.zip")
        if item["url"] != url or len(item["official_zip_sha256"]) != 64:
            raise SourceFailure("Binance manifest URL/hash changed")
    return rows


def _parse_bn_zip(body: bytes, month: str, terminal: bool) -> tuple[pd.DataFrame | float, dict]:
    name = f"BTCUSDT-1d-{month}.csv"
    try:
        with zipfile.ZipFile(BytesIO(body)) as archive:
            if archive.namelist() != [name]:
                raise SourceFailure(f"{month}: Binance ZIP member changed")
            raw = archive.read(name)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise SourceFailure(f"{month}: invalid Binance ZIP: {exc}") from exc
    unit = 1000000 if month >= "2025-01" else 1000
    unit_name = "us" if unit == 1000000 else "ms"
    try:
        if terminal:
            lines = raw.splitlines()
            if not lines:
                raise SourceFailure("terminal Binance ZIP empty")
            row = next(csv.reader([lines[0].decode("utf-8-sig")]))
            stamp = int(row[0])
            target = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()) * unit
            op = float(row[1])
            if len(row) != 12 or stamp != target or not math.isfinite(op) or op <= 0:
                raise SourceFailure("invalid terminal first-row open")
            return op, {"month": month, "terminal_first_open_only": True,
                        "timestamp_unit": unit_name}
        rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
        first = date.fromisoformat(month + "-01")
        next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        expected_count = (next_month - first).days
        if len(rows) != expected_count:
            raise SourceFailure(f"{month}: Binance missing/extra days")
        parsed = []
        for i, row in enumerate(rows):
            if len(row) != 12:
                raise SourceFailure(f"{month}: Binance CSV schema changed")
            stamp = int(row[0]); close_stamp = int(row[6])
            expected = int(datetime.combine(first + timedelta(days=i),
                                            datetime.min.time(), timezone.utc).timestamp()) * unit
            if stamp != expected or close_stamp != stamp + 86400 * unit - 1:
                raise SourceFailure(f"{month}: Binance timestamp grid invalid")
            op, hi, lo, cl, base_vol, quote_vol = [float(row[j]) for j in (1, 2, 3, 4, 5, 7)]
            if not all(math.isfinite(v) for v in (op, hi, lo, cl, base_vol, quote_vol)):
                raise SourceFailure(f"{month}: Binance nonfinite OHLCV")
            if lo <= 0 or hi < max(op, cl) or lo > min(op, cl) or base_vol < 0 or quote_vol < 0:
                raise SourceFailure(f"{month}: Binance invalid OHLCV")
            parsed.append((first + timedelta(days=i), op, hi, lo, cl, base_vol, quote_vol))
        dates = pd.date_range(first, next_month - timedelta(days=1), tz="UTC", freq="D")
        frame = pd.DataFrame([x[1:] for x in parsed], index=dates,
                             columns=["open", "high", "low", "close", "base_volume", "quote_volume"])
        return frame, {"month": month, "days": len(frame), "first": str(first),
                       "last": str(next_month - timedelta(days=1)), "timestamp_unit": unit_name}
    except SourceFailure:
        raise
    except (ValueError, TypeError, IndexError, UnicodeError, OverflowError) as exc:
        raise SourceFailure(f"{month}: invalid Binance price CSV: {exc}") from exc


def _load_bn_part(item: dict[str, str]) -> tuple[str, pd.DataFrame | float, dict]:
    month, url, pin = item["month"], item["url"], item["official_zip_sha256"]
    checksum = _fetch(url + ".CHECKSUM").decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-f]{64})\s+" + re.escape(url.rsplit("/", 1)[-1]), checksum)
    if match is None or match.group(1) != pin:
        raise SourceFailure(f"{month}: Binance checksum changed")
    body = _fetch(url)
    digest = sha256(body).hexdigest()
    if digest != pin:
        raise SourceFailure(f"{month}: Binance ZIP hash mismatch")
    data, audit = _parse_bn_zip(body, month, month == "2026-10")
    audit.update({"url": url, "sha256": digest, "bytes": len(body)})
    return month, data, audit


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, dict]:
    cb_manifest = _coinbase_manifest()
    bn_manifest = _binance_manifest()
    with ThreadPoolExecutor(max_workers=3) as pool:
        cb_parts = list(pool.map(_load_cb_part, cb_manifest))
    cb_frames = {}
    for pair in ("BTC-USD", "USDT-USD"):
        frames = [frame for p, frame, _audit in cb_parts if p == pair]
        joined = pd.concat(frames).sort_index()
        expected = pd.date_range(CB_START, LAST, tz="UTC", freq="D")
        if not joined.index.equals(expected):
            raise SourceFailure(f"Coinbase {pair} joined grid invalid")
        cb_frames[pair] = joined
    with ThreadPoolExecutor(max_workers=4) as pool:
        bn_parts = list(pool.map(_load_bn_part, bn_manifest))
    bn_parts.sort(key=lambda x: x[0])
    terminal = bn_parts[-1][1]
    if not isinstance(terminal, float):
        raise SourceFailure("Binance terminal open absent")
    frames = [x[1] for x in bn_parts[:-1]]
    if not all(isinstance(x, pd.DataFrame) for x in frames):
        raise SourceFailure("Binance nonterminal archive invalid")
    binance = pd.concat(frames)
    expected_bn = pd.date_range(BN_START, LAST, tz="UTC", freq="D")
    if not binance.index.equals(expected_bn):
        raise SourceFailure("Binance joined daily grid invalid")
    audit = {"coinbase": [x[2] for x in cb_parts], "binance": [x[2] for x in bn_parts],
             "first": str(CB_START), "last": str(LAST),
             "historical_source_vintages_unverified": True}
    return cb_frames["BTC-USD"], cb_frames["USDT-USD"], binance, terminal, audit
