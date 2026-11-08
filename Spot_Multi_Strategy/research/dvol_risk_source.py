"""Pinned Deribit DVOL and Binance spot histories for one sealed experiment."""

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
from zipfile import BadZipFile, ZipFile

import pandas as pd

from research.stablecoin_liquidity_source import _fetch


FIRST_PRICE = date(2021, 3, 1)
FIRST_DVOL = date(2021, 3, 24)
START = date(2021, 5, 1)
END = date(2026, 10, 1)
SYMBOLS = ("BTCUSDT", "ETHUSDT")
CURRENCIES = ("BTC", "ETH")
ROOT = Path(__file__).parent / "preregistrations"
DVOL_MANIFEST = ROOT / "2026-11-08_deribit_dvol_year_manifest.csv"
SPOT_MANIFEST = ROOT / "2026-11-08_deribit_dvol_spot_manifest.csv"
DVOL_FIELDS = ("currency", "year", "start_timestamp_ms", "end_timestamp_ms",
               "rows", "first_utc", "last_utc", "canonical_data_sha256")
SPOT_FIELDS = ("month", "market", "symbol", "dataset", "sha256", "days",
               "timestamp_unit")


class DvolSourceFailure(ValueError):
    pass


def _rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != fields:
                raise DvolSourceFailure(f"{path.name}: manifest schema changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DvolSourceFailure(f"{path.name}: manifest unavailable") from exc
    if any(set(row) != set(fields) for row in rows):
        raise DvolSourceFailure(f"{path.name}: malformed manifest row")
    return rows


def _expected_dvol() -> list[tuple[str, int, date, date]]:
    return [(c, y, FIRST_DVOL if y == 2021 else date(y, 1, 1),
             date(2026, 9, 30) if y == 2026 else date(y, 12, 31))
            for y in range(2021, 2027) for c in CURRENCIES]


def _dvol_year(row: dict[str, str]) -> tuple[str, pd.DataFrame, dict]:
    c, y = row["currency"], int(row["year"])
    first = FIRST_DVOL if y == 2021 else date(y, 1, 1)
    last = date(2026, 9, 30) if y == 2026 else date(y, 12, 31)
    start_ms = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int((datetime.combine(last + timedelta(days=1), datetime.min.time(),
                                   timezone.utc) - timedelta(milliseconds=1)).timestamp() * 1000)
    count = (last - first).days + 1
    if (row["start_timestamp_ms"] != str(start_ms) or row["end_timestamp_ms"] != str(end_ms)
            or row["first_utc"] != str(first) or row["last_utc"] != str(last)
            or row["rows"] != str(count)
            or not re.fullmatch(r"[0-9a-f]{64}", row["canonical_data_sha256"] or "")):
        raise DvolSourceFailure(f"{c} {y}: DVOL manifest specification changed")
    body = _fetch("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                  limit=1_000_000, params={"currency": c, "start_timestamp": start_ms,
                                           "end_timestamp": end_ms, "resolution": "1D"})
    try:
        payload = json.loads(body)
        result = payload["result"]
        if payload.get("error") is not None or result.get("continuation") is not None:
            raise DvolSourceFailure(f"{c} {y}: DVOL incomplete/error")
        data = result["data"]
        if not isinstance(data, list) or len(data) != count:
            raise DvolSourceFailure(f"{c} {y}: DVOL daily coverage changed")
        digest = sha256(json.dumps(data, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if digest != row["canonical_data_sha256"]:
            raise DvolSourceFailure(f"{c} {y}: DVOL canonical data revised: {digest}")
        values = []
        for i, item in enumerate(data):
            day = first + timedelta(days=i)
            stamp = int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp() * 1000)
            if (not isinstance(item, list) or len(item) != 5 or type(item[0]) is not int
                    or item[0] != stamp):
                raise DvolSourceFailure(f"{c} {day}: DVOL UTC candle invalid")
            nums = item[1:]
            if (not all(type(v) in (float, int) and math.isfinite(v) and v > 2
                        for v in nums) or nums[1] < max(nums[0], nums[2], nums[3])
                    or nums[2] > min(nums[0], nums[1], nums[3])):
                raise DvolSourceFailure(f"{c} {day}: DVOL units/OHLC invalid")
            values.append(nums)
        frame = pd.DataFrame(values, index=pd.date_range(first, last, tz="UTC", freq="D"),
                             columns=("open", "high", "low", "close"))
        return c, frame, {"currency": c, "year": y, "rows": count, "sha256": digest,
                          "first": str(first), "last": str(last)}
    except DvolSourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise DvolSourceFailure(f"{c} {y}: malformed DVOL source") from exc


def load_dvol() -> tuple[dict[str, pd.DataFrame], list[dict]]:
    rows = _rows(DVOL_MANIFEST, DVOL_FIELDS)
    expected = _expected_dvol()
    if [(r["currency"], int(r["year"]), date.fromisoformat(r["first_utc"]),
         date.fromisoformat(r["last_utc"])) for r in rows] != expected:
        raise DvolSourceFailure("DVOL manifest currency/year order changed")
    with ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(_dvol_year, rows))
    frames = {c: pd.concat([frame for symbol, frame, _ in parts if symbol == c])
              for c in CURRENCIES}
    expected_index = pd.date_range(FIRST_DVOL, END - timedelta(days=1), tz="UTC", freq="D")
    if any(not frame.index.equals(expected_index) for frame in frames.values()):
        raise DvolSourceFailure("DVOL full daily grid incomplete")
    return frames, [audit for _, _, audit in parts]


def _months() -> list[str]:
    return [f"{y}-{m:02d}" for y in range(2021, 2027) for m in range(1, 13)
            if (2021, 3) <= (y, m) <= (2026, 10)]


def _spot_month(row: dict[str, str]) -> tuple[str, str, pd.DataFrame | float, dict]:
    month, symbol, pin = row["month"], row["symbol"], row["sha256"]
    y, m = map(int, month.split("-"))
    n = monthrange(y, m)[1]
    unit = 1_000_000 if month >= "2025-01" else 1000
    if (row["market"] != "spot" or row["dataset"] != "1d" or row["days"] != str(n)
            or row["timestamp_unit"] != ("us" if unit == 1_000_000 else "ms")
            or not re.fullmatch(r"[0-9a-f]{64}", pin or "")):
        raise DvolSourceFailure(f"{symbol} {month}: spot manifest changed")
    name = f"{symbol}-1d-{month}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{name}"
    checksum = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", checksum)
    if match is None or match[1].lower() != pin or match[2] != name:
        raise DvolSourceFailure(f"{symbol} {month}: official spot checksum changed")
    body = _fetch(url, limit=150_000)
    if sha256(body).hexdigest() != pin:
        raise DvolSourceFailure(f"{symbol} {month}: spot ZIP revised")
    values: list[tuple[float, float, float, float, float]] = []
    first_stamp = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp()) * unit
    try:
        with ZipFile(BytesIO(body)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].flag_bits & 1 or members[0].file_size > 200_000):
                raise DvolSourceFailure(f"{symbol} {month}: unexpected spot ZIP member")
            with archive.open(members[0]) as stream:
                for i, fields in enumerate(csv.reader(TextIOWrapper(stream, encoding="utf-8-sig",
                                                                    newline=""), strict=True)):
                    stamp = first_stamp + i * 86_400 * unit
                    if (i >= n or len(fields) != 12 or fields[0] != str(stamp)
                            or fields[6] != str(stamp + 86_400 * unit - 1)):
                        raise DvolSourceFailure(f"{symbol} {month}: spot UTC day missing")
                    nums = [float(fields[k]) for k in (1, 2, 3, 4, 5, 7, 9, 10)]
                    op, hi, lo, cl, base, quote, taker_base, taker_quote = nums
                    if (not all(math.isfinite(x) for x in nums) or min(op, hi, lo, cl) <= 0
                            or hi < max(op, lo, cl) or lo > min(op, hi, cl)
                            or min(base, quote, taker_base, taker_quote) < 0
                            or taker_base > base or taker_quote > quote or int(fields[8]) < 0):
                        raise DvolSourceFailure(f"{symbol} {month}: invalid spot OHLCV")
                    values.append((op, hi, lo, cl, quote))
    except DvolSourceFailure:
        raise
    except (BadZipFile, OSError, UnicodeError, csv.Error, ValueError, IndexError) as exc:
        raise DvolSourceFailure(f"{symbol} {month}: malformed spot ZIP") from exc
    if len(values) != n:
        raise DvolSourceFailure(f"{symbol} {month}: incomplete spot archive")
    if month == "2026-10":
        value: pd.DataFrame | float = values[0][0]
    else:
        value = pd.DataFrame(values, index=pd.date_range(f"{month}-01", periods=n,
                                                          tz="UTC", freq="D"),
                             columns=("open", "high", "low", "close", "quote_volume"))
    return month, symbol, value, {"month": month, "symbol": symbol, "sha256": pin,
                                  "bytes": len(body)}


def load_spot() -> tuple[dict[str, pd.DataFrame], dict[str, float], list[dict]]:
    rows = _rows(SPOT_MANIFEST, SPOT_FIELDS)
    expected = [(month, symbol) for month in _months() for symbol in SYMBOLS]
    if [(r["month"], r["symbol"]) for r in rows] != expected:
        raise DvolSourceFailure("spot manifest month/symbol order changed")
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(_spot_month, rows))
    prices = {s: pd.concat([part for month, symbol, part, _ in parts
                            if symbol == s and month != "2026-10"])
              for s in SYMBOLS}
    terminal = {symbol: float(part) for month, symbol, part, _ in parts
                if month == "2026-10"}
    expected_index = pd.date_range(FIRST_PRICE, END - timedelta(days=1), tz="UTC", freq="D")
    if (set(terminal) != set(SYMBOLS) or
            any(not frame.index.equals(expected_index) for frame in prices.values())):
        raise DvolSourceFailure("spot full daily grid incomplete")
    return prices, terminal, [audit for _, _, _, audit in parts]
