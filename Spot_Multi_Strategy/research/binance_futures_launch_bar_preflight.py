"""Checksum and timestamp audit of launch-window 5m archives, without returns.

The only derived values are source validity flags, row counts, UTC timestamps,
and SHA-256 digests. No prices or strategy metrics are retained or printed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import time
from zipfile import ZipFile, BadZipFile

import requests


HERE = Path(__file__).parent
PRESENCE = HERE / "preregistrations/2026-11-08_binance_futures_launch_archive_presence.json"
OUTPUT = HERE / "preregistrations/2026-11-08_binance_futures_launch_bar_integrity.json"
ROOT = "https://data.binance.vision/data/futures/um/daily/klines"
MAX_ZIP = 2_000_000
MAX_CSV = 8_000_000
STEP_MS = 300_000


class LaunchBarFailure(ValueError):
    pass


def _get(url: str, limit: int) -> bytes:
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(10, 40))
            if response.status_code != 200 or len(response.content) > limit:
                raise LaunchBarFailure(f"archive HTTP/size failure {response.status_code}: {url}")
            return response.content
        except requests.RequestException as exc:
            if attempt == 2:
                raise LaunchBarFailure(f"archive transport failure: {url}") from exc
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def _number(text: str, *, positive: bool = False) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise LaunchBarFailure("nonnumeric kline field") from exc
    if not value.is_finite() or (value <= 0 if positive else value < 0):
        raise LaunchBarFailure("invalid kline numeric field")
    return value


def _day(symbol: str, day: str) -> tuple[set[int], dict]:
    name = f"{symbol}-5m-{day}.zip"
    url = f"{ROOT}/{symbol}/5m/{name}"
    checksum_bytes = _get(url + ".CHECKSUM", 300)
    try:
        statement = checksum_bytes.decode("ascii").strip()
    except UnicodeError as exc:
        raise LaunchBarFailure("non-ASCII checksum") from exc
    match = re.fullmatch(r"([0-9a-fA-F]{64})\s+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise LaunchBarFailure("official checksum malformed")
    zip_bytes = _get(url, MAX_ZIP)
    digest = sha256(zip_bytes).hexdigest()
    if digest != match.group(1).lower():
        raise LaunchBarFailure("official checksum mismatch")
    try:
        with ZipFile(io.BytesIO(zip_bytes)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != name[:-4] + ".csv":
                raise LaunchBarFailure("unexpected ZIP members")
            if members[0].file_size > MAX_CSV or members[0].flag_bits & 1:
                raise LaunchBarFailure("unsafe ZIP member")
            with archive.open(members[0]) as member:
                rows = list(csv.reader(io.TextIOWrapper(member, encoding="utf-8-sig")))
    except (BadZipFile, UnicodeError, csv.Error) as exc:
        raise LaunchBarFailure("invalid kline ZIP/CSV") from exc
    if rows and rows[0][0] == "open_time":
        rows.pop(0)
    if not rows:
        raise LaunchBarFailure("empty kline archive")
    day_start = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)
    opened = []
    for row in rows:
        if len(row) != 12 or not row[0].isdigit() or not row[6].isdigit():
            raise LaunchBarFailure("invalid kline column count or timestamp")
        t, close_t = int(row[0]), int(row[6])
        if (t < day_start or t >= day_start + 86_400_000
                or (t - day_start) % STEP_MS or close_t != t + STEP_MS - 1):
            raise LaunchBarFailure("off-grid kline timestamp")
        op, hi, lo, cl = (_number(row[i], positive=True) for i in (1, 2, 3, 4))
        if hi < max(op, cl) or lo > min(op, cl):
            raise LaunchBarFailure("OHLC ordering violation")
        base_vol, quote_vol = _number(row[5]), _number(row[7])
        taker_base, taker_quote = _number(row[9]), _number(row[10])
        if (not row[8].isdigit() or taker_base > base_vol
                or taker_quote > quote_vol):
            raise LaunchBarFailure("volume or trade-count violation")
        opened.append(t)
    if any(b - a != STEP_MS for a, b in zip(opened, opened[1:])):
        raise LaunchBarFailure("kline archive has duplicate or missing internal bars")
    return set(opened), {
        "url": url, "sha256": digest, "checksum_sha256": sha256(checksum_bytes).hexdigest(),
        "rows": len(opened), "first_utc": datetime.fromtimestamp(opened[0] / 1000, timezone.utc).isoformat(),
        "last_utc": datetime.fromtimestamp(opened[-1] / 1000, timezone.utc).isoformat(),
    }


def _event(row: dict) -> dict:
    launch = datetime.fromisoformat(row["launch_utc"])
    if launch.utcoffset() != timedelta(0) or launch.second or launch.microsecond or launch.minute % 5:
        raise LaunchBarFailure("launch is not on a five-minute UTC boundary")
    symbol = row["contract_symbol"]
    first_bars, first = _day(symbol, launch.date().isoformat())
    second_bars, second = _day(symbol, (launch + timedelta(days=1)).date().isoformat())
    event_ms = int(launch.timestamp() * 1000)
    required = range(event_ms + 30 * 60_000, event_ms + 24 * 60 * 60_000 + 1, STEP_MS)
    all_bars = first_bars | second_bars
    missing = sum(t not in all_bars for t in required)
    return {
        "article_code": row["article_code"], "contract_symbol": symbol,
        "launch_utc": row["launch_utc"],
        "first_bar_matches_scheduled_launch": min(first_bars) == event_ms,
        "required_window_bar_count": len(required),
        "missing_required_window_bars": missing,
        "launch_day": first, "next_day": second,
    }


def inventory() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("launch bar integrity manifest already exists")
    source_bytes = PRESENCE.read_bytes()
    source = json.loads(source_bytes)
    events = [row for row in source["rows"] if row["http_status"]["prior_day"] == 404
              and row["http_status"]["launch_day"] == 200
              and row["http_status"]["next_day"] == 200]
    if len(events) != source["event_count"] - 1:
        raise LaunchBarFailure("unexpected source presence attrition")
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_event, event): event for event in events}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
    rows.sort(key=lambda row: (row["launch_utc"], row["article_code"]))
    result = {
        "source_manifest_sha256": sha256(source_bytes).hexdigest(),
        "events": len(rows),
        "scheduled_first_bar_mismatches": sum(not r["first_bar_matches_scheduled_launch"] for r in rows),
        "missing_required_window_bars": sum(r["missing_required_window_bars"] for r in rows),
        "rows": rows,
        "source_integrity_only_no_strategy_return_calculated": True,
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


if __name__ == "__main__":
    r = inventory()
    print(json.dumps({key: r[key] for key in ("events", "scheduled_first_bar_mismatches",
                                               "missing_required_window_bars")}, sort_keys=True))
