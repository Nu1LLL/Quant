"""Frozen source audit for the ETH DeFi TVL flow experiment; no trading code."""

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
MANIFEST = ROOT / "preregistrations/2026-11-08_eth_defi_tvl_spot_manifest.csv"
TVL_URL = "https://api.llama.fi/v2/historicalChainTvl/Ethereum"
TVL_SHA = "46e684d5efea4f44a8e85168372a64c0b70b6c790617f8387e034ed6187d628d"
TVL_FIRST = date(2020, 11, 1)
TVL_LAST = date(2025, 12, 31)
PRICE_FIRST = date(2020, 11, 1)
PRICE_LAST = date(2025, 12, 31)
TERMINAL = date(2026, 1, 1)


class SourceFailure(ValueError):
    pass


def _fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=35, headers={
                "User-Agent": "MiniMedallionResearch/1.0 (public data audit)"})
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(attempt + 1)
    raise SourceFailure(f"failed source request: {url}: {last}")


def parse_tvl(body: bytes) -> dict[date, float]:
    try:
        rows = json.loads(body)
        if not isinstance(rows, list):
            raise SourceFailure("TVL response is not a list")
        start = int(datetime.combine(TVL_FIRST, datetime.min.time(), timezone.utc).timestamp())
        end = int(datetime.combine(TVL_LAST, datetime.min.time(), timezone.utc).timestamp())
        selected = [r for r in rows if isinstance(r, dict) and isinstance(r.get("date"), int)
                    and start <= r["date"] <= end]
        digest = sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != TVL_SHA:
            raise SourceFailure(f"TVL historical subset digest changed: {digest}")
        expected = (TVL_LAST - TVL_FIRST).days + 1
        if len(selected) != expected:
            raise SourceFailure(f"TVL count {len(selected)} != {expected}")
        output: dict[date, float] = {}
        for i, row in enumerate(selected):
            timestamp = row["date"]
            day = datetime.fromtimestamp(timestamp, timezone.utc).date()
            if timestamp != start + 86400 * i or day in output:
                raise SourceFailure(f"TVL date grid error at {day}")
            value = float(row["tvl"])
            if not math.isfinite(value) or value <= 0:
                raise SourceFailure(f"invalid TVL value at {day}")
            output[day] = value
        return output
    except SourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure(f"invalid TVL response: {exc}") from exc


def _parse_spot_zip(body: bytes, month: str, terminal: bool) -> tuple[pd.DataFrame | float, dict]:
    name = f"ETHUSDT-1d-{month}.csv"
    try:
        with zipfile.ZipFile(BytesIO(body)) as zf:
            if zf.namelist() != [name]:
                raise SourceFailure(f"{month}: unexpected ZIP members {zf.namelist()}")
            raw = zf.read(name)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise SourceFailure(f"{month}: invalid ZIP: {exc}") from exc
    unit = "us" if month >= "2025-01" else "ms"
    multiplier = 1000000 if unit == "us" else 1000
    if terminal:
        lines = raw.splitlines()
        if not lines:
            raise SourceFailure("terminal archive empty")
        row = next(csv.reader([lines[0].decode("utf-8-sig")]))
        if len(row) != 12:
            raise SourceFailure("terminal first row has wrong schema")
        stamp = int(row[0]); expected = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * multiplier
        value = float(row[1])
        if stamp != expected or not math.isfinite(value) or value <= 0:
            raise SourceFailure("terminal first row timestamp/open invalid")
        return value, {"month": month, "terminal_open_only": True, "timestamp_unit": unit}
    try:
        frame = pd.read_csv(BytesIO(raw), header=None)
        if frame.shape[1] != 12:
            raise SourceFailure(f"{month}: expected 12 CSV fields")
        stamps = pd.to_numeric(frame[0], errors="raise").astype("int64").to_numpy()
        close_stamps = pd.to_numeric(frame[6], errors="raise").astype("int64").to_numpy()
        if ((close_stamps - stamps) != 86400 * multiplier - 1).any():
            raise SourceFailure(f"{month}: invalid daily close timestamp")
        dates = pd.to_datetime(stamps, unit=unit, utc=True)
        first = date.fromisoformat(month + "-01")
        next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        expected_dates = pd.date_range(first, next_month - timedelta(days=1), tz="UTC", freq="D")
        if not dates.equals(expected_dates):
            raise SourceFailure(f"{month}: missing, duplicate or extra daily bar")
        prices = frame[[1, 2, 3, 4, 5]].apply(pd.to_numeric, errors="raise").astype("float64")
        prices.columns = ["open", "high", "low", "close", "volume"]
        prices.index = dates
        if not prices.map(math.isfinite).all().all():
            raise SourceFailure(f"{month}: nonfinite OHLCV")
        if (prices[["open", "high", "low", "close"]] <= 0).any().any() or (prices["volume"] < 0).any():
            raise SourceFailure(f"{month}: invalid OHLCV sign")
        if ((prices["low"] > prices[["open", "close"]].min(axis=1)) |
            (prices["high"] < prices[["open", "close"]].max(axis=1)) |
            (prices["low"] > prices["high"])).any():
            raise SourceFailure(f"{month}: invalid OHLC ordering")
        return prices, {"month": month, "rows": len(prices), "first": str(dates[0]),
                        "last": str(dates[-1]), "timestamp_unit": unit}
    except SourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure(f"{month}: invalid price CSV: {exc}") from exc


def _load_month(record: dict[str, str]) -> tuple[str, pd.DataFrame | float, dict]:
    month, url, pin = record["month"], record["url"], record["official_zip_sha256"]
    checksum = _fetch(url + ".CHECKSUM").decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-f]{64})\s+" + re.escape(url.rsplit("/", 1)[-1]), checksum)
    if match is None or match.group(1) != pin:
        raise SourceFailure(f"{month}: official checksum changed")
    body = _fetch(url)
    digest = sha256(body).hexdigest()
    if digest != pin:
        raise SourceFailure(f"{month}: ZIP SHA mismatch")
    data, audit = _parse_spot_zip(body, month, month == "2026-01")
    audit.update({"url": url, "sha256": digest, "bytes": len(body)})
    return month, data, audit


def load_sources() -> tuple[dict[date, float], pd.DataFrame, float, dict]:
    tvl_body = _fetch(TVL_URL)
    tvl = parse_tvl(tvl_body)
    with MANIFEST.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    expected_months = pd.period_range("2020-11", "2026-01", freq="M").astype(str).tolist()
    if [x["month"] for x in manifest] != expected_months:
        raise SourceFailure("spot source manifest month grid changed")
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, manifest))
    parts.sort(key=lambda x: x[0])
    terminal = parts[-1][1]
    if not isinstance(terminal, float):
        raise SourceFailure("missing terminal open")
    frames = [x[1] for x in parts[:-1]]
    if not all(isinstance(x, pd.DataFrame) for x in frames):
        raise SourceFailure("nonterminal archive malformed")
    prices = pd.concat(frames)
    expected = pd.date_range(PRICE_FIRST, PRICE_LAST, tz="UTC", freq="D")
    if not prices.index.equals(expected):
        raise SourceFailure("joined ETH price grid invalid")
    audit = {"tvl": {"url": TVL_URL, "subset_sha256": TVL_SHA,
                     "response_sha256": sha256(tvl_body).hexdigest(),
                     "days": len(tvl), "first": str(TVL_FIRST), "last": str(TVL_LAST),
                     "backfilled_not_point_in_time": True},
             "spot": [x[2] for x in parts]}
    return tvl, prices, terminal, audit
