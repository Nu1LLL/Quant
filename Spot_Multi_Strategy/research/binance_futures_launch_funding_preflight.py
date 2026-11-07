"""Validate launch-window funding sources without retaining funding values.

Uses official checksum-verified monthly ZIPs. Only the two archive gaps fixed
by the frozen presence manifest may use Binance's historical REST endpoint.
No trading rule, return, or PnL is calculated.
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
BARS = HERE / "preregistrations/2026-11-08_binance_futures_launch_bar_integrity.json"
OUTPUT = HERE / "preregistrations/2026-11-08_binance_futures_launch_funding_integrity.json"
ARCHIVE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
REST = "https://fapi.binance.com/fapi/v1/fundingRate"
FALLBACK_SYMBOLS = frozenset({"ICPUSDT", "BNXUSDT"})


class FundingPreflightFailure(ValueError):
    pass


def _get(url: str, *, params: dict | None = None, max_bytes: int) -> bytes:
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=(10, 40))
            if response.status_code != 200 or len(response.content) > max_bytes:
                raise FundingPreflightFailure(f"funding HTTP/size failure {response.status_code}: {url}")
            return response.content
        except requests.RequestException as exc:
            if attempt == 2:
                raise FundingPreflightFailure(f"funding transport failure: {url}") from exc
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def _rate(text: str) -> None:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise FundingPreflightFailure("nonnumeric funding rate") from exc
    if not value.is_finite() or abs(value) > 1:
        raise FundingPreflightFailure("invalid funding rate")


def _archive(symbol: str, month: str) -> tuple[list[int], dict]:
    name = f"{symbol}-fundingRate-{month}.zip"
    url = f"{ARCHIVE}/{symbol}/{name}"
    checksum_bytes = _get(url + ".CHECKSUM", max_bytes=300)
    match = re.fullmatch(r"([0-9a-fA-F]{64})\s+([^\s]+)",
                         checksum_bytes.decode("ascii").strip())
    if match is None or match.group(2) != name:
        raise FundingPreflightFailure("malformed official funding checksum")
    archive_bytes = _get(url, max_bytes=500_000)
    digest = sha256(archive_bytes).hexdigest()
    if digest != match.group(1).lower():
        raise FundingPreflightFailure("funding checksum mismatch")
    try:
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].file_size > 2_000_000 or members[0].flag_bits & 1):
                raise FundingPreflightFailure("unsafe funding ZIP member")
            with archive.open(members[0]) as member:
                rows = list(csv.reader(io.TextIOWrapper(member, encoding="utf-8-sig")))
    except (BadZipFile, UnicodeError, csv.Error) as exc:
        raise FundingPreflightFailure("invalid funding ZIP/CSV") from exc
    if not rows or rows.pop(0) != ["calc_time", "funding_interval_hours", "last_funding_rate"]:
        raise FundingPreflightFailure("unexpected funding CSV header")
    times = []
    for row in rows:
        if (len(row) != 3 or not row[0].isdigit() or not row[1].isdigit()
                or not 1 <= int(row[1]) <= 24):
            raise FundingPreflightFailure("invalid funding CSV fields")
        _rate(row[2])
        t = int(row[0])
        if datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m") != month:
            raise FundingPreflightFailure("funding timestamp outside archive month")
        times.append(t)
    if not times or times != sorted(set(times)):
        raise FundingPreflightFailure("funding timestamps empty, unsorted or duplicated")
    return times, {"source": "checksum_zip", "url": url, "sha256": digest,
                   "checksum_sha256": sha256(checksum_bytes).hexdigest(),
                   "rows": len(times)}


def _rest(symbol: str, start: datetime, end: datetime) -> tuple[list[int], dict]:
    params = {"symbol": symbol, "startTime": int(start.timestamp() * 1000),
              "endTime": int(end.timestamp() * 1000), "limit": 1000}
    payload = _get(REST, params=params, max_bytes=100_000)
    data = json.loads(payload)
    if not isinstance(data, list):
        raise FundingPreflightFailure("unexpected REST funding response")
    times = []
    for row in data:
        if (not isinstance(row, dict) or row.get("symbol") != symbol
                or type(row.get("fundingTime")) is not int):
            raise FundingPreflightFailure("invalid REST funding identity/timestamp")
        _rate(row.get("fundingRate", ""))
        times.append(row["fundingTime"])
    if times != sorted(set(times)):
        raise FundingPreflightFailure("REST funding timestamps duplicate or unsorted")
    return times, {"source": "historical_rest_gap_fallback", "url": REST,
                   "request_params": params, "response_sha256": sha256(payload).hexdigest(),
                   "rows": len(times)}


def _event(row: dict, presence: dict) -> dict:
    symbol = row["contract_symbol"]
    scheduled = datetime.fromisoformat(row["launch_utc"])
    first_bar = datetime.fromisoformat(row["launch_day"]["first_utc"])
    effective = max(scheduled, first_bar)
    entry, exit_time = effective + timedelta(minutes=30), effective + timedelta(hours=24)
    months = sorted({entry.strftime("%Y-%m"), exit_time.strftime("%Y-%m")})
    times, sources = [], []
    for month in months:
        status = presence["http_status"].get("funding_" + month)
        if status == 200:
            t, source = _archive(symbol, month)
        elif status == 404 and symbol in FALLBACK_SYMBOLS:
            t, source = _rest(symbol, entry, exit_time)
        else:
            raise FundingPreflightFailure(f"unplanned funding archive gap: {symbol} {month}")
        times.extend(t)
        sources.append(source)
    if len(times) != len(set(times)):
        raise FundingPreflightFailure("funding events overlap across sources")
    held = [t for t in times if int(entry.timestamp() * 1000) <= t < int(exit_time.timestamp() * 1000)]
    return {"article_code": row["article_code"], "contract_symbol": symbol,
            "effective_launch_utc": effective.isoformat(),
            "entry_utc": entry.isoformat(), "exit_utc": exit_time.isoformat(),
            "funding_events_during_hold": len(held), "sources": sources}


def inventory() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("funding integrity manifest already exists")
    presence_bytes, bar_bytes = PRESENCE.read_bytes(), BARS.read_bytes()
    p, b = json.loads(presence_bytes), json.loads(bar_bytes)
    lookup = {row["article_code"]: row for row in p["rows"]}
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_event, row, lookup[row["article_code"]]): row
                   for row in b["rows"]}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
    rows.sort(key=lambda row: (row["effective_launch_utc"], row["article_code"]))
    result = {"presence_sha256": sha256(presence_bytes).hexdigest(),
              "bar_integrity_sha256": sha256(bar_bytes).hexdigest(),
              "events": len(rows),
              "archive_sources": sum(s["source"] == "checksum_zip" for r in rows for s in r["sources"]),
              "rest_fallback_sources": sum(s["source"] == "historical_rest_gap_fallback"
                                           for r in rows for s in r["sources"]),
              "zero_funding_event_holds": sum(r["funding_events_during_hold"] == 0 for r in rows),
              "rows": rows,
              "source_integrity_only_no_strategy_return_calculated": True}
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


if __name__ == "__main__":
    r = inventory()
    print(json.dumps({key: r[key] for key in
                      ("events", "archive_sources", "rest_fallback_sources",
                       "zero_funding_event_holds")}, sort_keys=True))
