"""Pinned stablecoin circulation and Binance spot daily archives, without signals."""

from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import json
import math
from pathlib import Path
import re
import time
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests


FIRST = date(2020, 12, 1)
LAST = date(2025, 12, 31)
TERMINAL = date(2026, 1, 1)
SYMBOLS = ("BNBUSDT", "BTCUSDT", "ETHUSDT")
STABLECOINS = (
    (1, "USDT", "b8efdb975611fcf6c1526899fa23fb25dab7e99e4edf2f09fe2620eee7147b88"),
    (2, "USDC", "a983863adf1c2c1426765c35a00f2d1236966f54f3bc1711257fa16f9c7b6858"),
)
MANIFEST = (Path(__file__).parent /
            "preregistrations/2026-11-08_stablecoin_liquidity_spot_1d_manifest.csv")
FIELDS = ("month", "market", "symbol", "dataset", "sha256", "days", "timestamp_unit")


class LiquiditySourceFailure(ValueError):
    pass


def _fetch(url: str, *, limit: int, params: dict | None = None) -> bytes:
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, params=params,
                                    headers={"User-Agent": "MiniMedallionResearch/1.0"},
                                    timeout=35)
            response.raise_for_status()
            if len(response.content) > limit:
                raise LiquiditySourceFailure(f"source response oversized: {url}")
            return response.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(attempt + 1)
    raise LiquiditySourceFailure(f"source unavailable: {url}: {last}")


def parse_stablecoin(body: bytes, coin: tuple[int, str, str]
                     ) -> tuple[list[float], dict]:
    stable_id, symbol, pin = coin
    try:
        chart = json.loads(body)
        if not isinstance(chart, list):
            raise LiquiditySourceFailure(f"{symbol}: stablecoin chart absent")
        selected: list[list] = []
        for row in chart:
            if not isinstance(row, dict) or not {"date", "totalCirculating", "totalCirculatingUSD"} <= set(row):
                raise LiquiditySourceFailure(f"{symbol}: chart row schema changed")
            stamp = int(row["date"])
            day = datetime.fromtimestamp(stamp, timezone.utc).date()
            if FIRST <= day <= LAST:
                total = row["totalCirculating"]
                if not isinstance(total, dict) or "peggedUSD" not in total:
                    raise LiquiditySourceFailure(f"{symbol}: USD circulating units absent")
                selected.append([stamp, total["peggedUSD"]])
        if len(selected) != (LAST - FIRST).days + 1:
            raise LiquiditySourceFailure(f"{symbol}: incomplete circulation history")
        canonical = json.dumps(selected, separators=(",", ":"), ensure_ascii=False,
                               allow_nan=False).encode()
        digest = sha256(canonical).hexdigest()
        if digest != pin:
            raise LiquiditySourceFailure(f"{symbol}: historic circulation changed: {digest}")
        values = []
        for i, (stamp, raw) in enumerate(selected):
            day = FIRST + timedelta(days=i)
            if stamp != int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp()):
                raise LiquiditySourceFailure(f"{symbol}: duplicate/missing UTC day {day}")
            if type(raw) not in (int, float) or not math.isfinite(raw) or raw <= 0:
                raise LiquiditySourceFailure(f"{symbol}: nonpositive/nonfinite units on {day}")
            values.append(float(raw))
        return values, {"stablecoin_id": stable_id, "symbol": symbol,
                        "selected_subset_sha256": digest, "days": len(values),
                        "first": str(FIRST), "last": str(LAST),
                        "response_bytes": len(body),
                        "first_published_daily_vintage_unavailable": True}
    except LiquiditySourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise LiquiditySourceFailure(f"{symbol}: invalid stablecoin chart: {exc}") from exc


def _load_stablecoin(coin: tuple[int, str, str]):
    body = _fetch("https://stablecoins.llama.fi/stablecoincharts/all", limit=2_000_000,
                  params={"stablecoin": coin[0]})
    return parse_stablecoin(body, coin)


def load_stablecoins() -> tuple[pd.DataFrame, list[dict]]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        parts = list(pool.map(_load_stablecoin, STABLECOINS))
    frame = pd.DataFrame({coin[1]: part[0] for coin, part in zip(STABLECOINS, parts)},
                         index=pd.date_range(FIRST, LAST, tz="UTC", freq="D"))
    return frame, [part[1] for part in parts]


def _months() -> list[str]:
    return [f"{y}-{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
            if (2020, 12) <= (y, m) <= (2026, 1)]


def _manifest() -> list[dict[str, str]]:
    try:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise LiquiditySourceFailure("spot manifest schema changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise LiquiditySourceFailure("spot manifest unavailable") from exc
    expected = [(month, symbol) for month in _months() for symbol in SYMBOLS]
    if len(rows) != len(expected):
        raise LiquiditySourceFailure("spot manifest count changed")
    for row, (month, symbol) in zip(rows, expected):
        y, m = map(int, month.split("-"))
        unit = "us" if month >= "2025-01" else "ms"
        if (set(row) != set(FIELDS) or row["month"] != month or row["symbol"] != symbol
                or row["market"] != "spot" or row["dataset"] != "1d"
                or row["timestamp_unit"] != unit
                or row["days"] != str(monthrange(y, m)[1])
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"] or "")):
            raise LiquiditySourceFailure(f"spot manifest row changed: {month} {symbol}")
    return rows


def _load_month(row: dict[str, str]) -> tuple[str, str, pd.DataFrame | float, dict]:
    month, symbol, pin = row["month"], row["symbol"], row["sha256"]
    name = f"{symbol}-1d-{month}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{name}"
    checksum = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", checksum)
    if match is None or match.group(2) != name or match.group(1).lower() != pin:
        raise LiquiditySourceFailure(f"{month} {symbol}: official checksum changed")
    body = _fetch(url, limit=100_000)
    digest = sha256(body).hexdigest()
    if digest != pin:
        raise LiquiditySourceFailure(f"{month} {symbol}: archive SHA changed")
    y, m = map(int, month.split("-"))
    unit = 1_000_000 if month >= "2025-01" else 1_000
    stamp = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp()) * unit
    expected_count = monthrange(y, m)[1]
    values = []
    try:
        with ZipFile(BytesIO(body)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].flag_bits & 1 or members[0].file_size > 200_000):
                raise LiquiditySourceFailure(f"{month} {symbol}: unexpected ZIP member")
            with archive.open(members[0]) as raw:
                for fields in csv.reader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True):
                    if (len(fields) != 12 or fields[0] != str(stamp)
                            or fields[6] != str(stamp + 86_400 * unit - 1)):
                        raise LiquiditySourceFailure(f"{month} {symbol}: missing/bad UTC day")
                    if month == "2026-01":
                        if not values:
                            opening = float(fields[1])
                            if not math.isfinite(opening) or opening <= 0:
                                raise LiquiditySourceFailure(f"{symbol}: bad terminal opening")
                            values.append(opening)
                    else:
                        nums = [float(fields[k]) for k in (1, 2, 3, 4, 5, 7, 9, 10)]
                        op, high, low, close, base, quote, taker_base, taker_quote = nums
                        trades = int(fields[8])
                        if (not all(math.isfinite(x) for x in nums) or min(op, high, low, close) <= 0
                                or high < max(op, low, close) or low > min(op, high, close)
                                or min(base, quote, taker_base, taker_quote) < 0
                                or taker_base > base or taker_quote > quote or trades < 0):
                            raise LiquiditySourceFailure(f"{month} {symbol}: invalid OHLCV")
                        values.append((op, high, low, close, quote))
                    stamp += 86_400 * unit
    except LiquiditySourceFailure:
        raise
    except (BadZipFile, OSError, UnicodeError, csv.Error, ValueError, IndexError, OverflowError) as exc:
        raise LiquiditySourceFailure(f"{month} {symbol}: malformed archive: {exc}") from exc
    if (month == "2026-01" and len(values) != 1) or (month != "2026-01" and len(values) != expected_count):
        raise LiquiditySourceFailure(f"{month} {symbol}: incomplete rows")
    if month == "2026-01":
        result: pd.DataFrame | float = float(values[0])
    else:
        index = pd.date_range(f"{month}-01", periods=expected_count, tz="UTC", freq="D")
        result = pd.DataFrame(values, index=index,
                              columns=("open", "high", "low", "close", "quote_volume"))
    return month, symbol, result, {"month": month, "symbol": symbol,
                                   "sha256": digest, "bytes": len(body)}


def load_spot() -> tuple[dict[str, pd.DataFrame], dict[str, float], list[dict]]:
    rows = _manifest()
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(_load_month, rows))
    frames: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in SYMBOLS}
    terminal: dict[str, float] = {}
    for month, symbol, value, _ in parts:
        if month == "2026-01":
            if not isinstance(value, float):
                raise LiquiditySourceFailure("terminal spot type invalid")
            terminal[symbol] = value
        else:
            if not isinstance(value, pd.DataFrame):
                raise LiquiditySourceFailure("spot historical type invalid")
            frames[symbol].append(value)
    expected = pd.date_range(FIRST, LAST, tz="UTC", freq="D")
    merged = {symbol: pd.concat(frames[symbol]) for symbol in SYMBOLS}
    if set(terminal) != set(SYMBOLS) or any(not frame.index.equals(expected)
                                           for frame in merged.values()):
        raise LiquiditySourceFailure("assembled spot grid invalid")
    return merged, terminal, [part[3] for part in parts]
