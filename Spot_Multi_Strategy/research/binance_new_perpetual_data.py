"""Reopen only preregistered Binance launch archives after source preflight.

This loader is read-only. Every official ZIP, checksum, REST response and BTC
factor archive is checked against a frozen pre-result hash before use.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile, BadZipFile

import requests

from .binance_futures_launch_bar_preflight import _get as archive_get
from .binance_futures_launch_funding_preflight import REST


HERE = Path(__file__).parent
PREREG = HERE / "preregistrations"
BAR_MANIFEST = PREREG / "2026-11-08_binance_futures_launch_bar_integrity.json"
FUNDING_MANIFEST = PREREG / "2026-11-08_binance_futures_launch_funding_integrity.json"
BTC_MANIFEST = PREREG / "2026-11-07_leadlag_5m_timestamp_manifest.csv"
BTC_ROOT = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/5m"


class LaunchDataFailure(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Bar:
    time: datetime
    opening: float
    high: float
    low: float
    close: float
    quote_volume: float


@dataclass(frozen=True, slots=True)
class Funding:
    time: datetime
    rate: float


@dataclass(frozen=True, slots=True)
class EventTape:
    article_code: str
    symbol: str
    scheduled: datetime
    effective: datetime
    entry: datetime
    exit: datetime
    bars: tuple[Bar, ...]  # 5m opens from effective through scheduled exit
    funding: tuple[Funding, ...]


@dataclass(frozen=True, slots=True)
class LaunchTape:
    events: tuple[EventTape, ...]
    btc_close: dict[date, float]
    source_hashes: dict[str, str]


def _verified_zip(url: str, pin: str, checksum_pin: str | None,
                  max_zip: int) -> bytes:
    name = url.rsplit("/", 1)[-1]
    checksum = archive_get(url + ".CHECKSUM", 300)
    if checksum_pin is not None and sha256(checksum).hexdigest() != checksum_pin:
        raise LaunchDataFailure(f"changed checksum statement: {name}")
    try:
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+([^\s]+)",
                             checksum.decode("ascii").strip())
    except UnicodeError as exc:
        raise LaunchDataFailure(f"non-ASCII checksum: {name}") from exc
    if match is None or match.group(2) != name or match.group(1).lower() != pin:
        raise LaunchDataFailure(f"checksum differs from pin: {name}")
    body = archive_get(url, max_zip)
    if sha256(body).hexdigest() != pin:
        raise LaunchDataFailure(f"archive differs from pin: {name}")
    return body


def _csv_rows(body: bytes, name: str, max_csv: int) -> list[list[str]]:
    try:
        with ZipFile(io.BytesIO(body)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].file_size > max_csv or members[0].flag_bits & 1):
                raise LaunchDataFailure(f"unsafe ZIP member: {name}")
            with archive.open(members[0]) as member:
                return list(csv.reader(io.TextIOWrapper(member, encoding="utf-8-sig")))
    except (BadZipFile, UnicodeError, csv.Error) as exc:
        raise LaunchDataFailure(f"invalid ZIP/CSV: {name}") from exc


def _bar_rows(pin: dict) -> list[Bar]:
    url = pin["url"]
    body = _verified_zip(url, pin["sha256"], pin["checksum_sha256"], 2_000_000)
    rows = _csv_rows(body, url.rsplit("/", 1)[-1], 8_000_000)
    if rows and rows[0][0] == "open_time":
        rows.pop(0)
    if len(rows) != pin["rows"]:
        raise LaunchDataFailure("daily bar count changed")
    bars = []
    for row in rows:
        if len(row) != 12:
            raise LaunchDataFailure("daily bar columns changed")
        try:
            values = [float(row[i]) for i in (1, 2, 3, 4, 7)]
            opened = int(row[0])
        except ValueError as exc:
            raise LaunchDataFailure("nonnumeric daily bar") from exc
        op, hi, lo, close, quote = values
        if (not all(math.isfinite(x) for x in values) or min(op, hi, lo, close) <= 0
                or quote < 0 or hi < max(op, close) or lo > min(op, close)):
            raise LaunchDataFailure("invalid OHLC/volume in pinned archive")
        bars.append(Bar(datetime.fromtimestamp(opened / 1000, timezone.utc),
                        op, hi, lo, close, quote))
    if (not bars or bars[0].time.isoformat() != pin["first_utc"]
            or bars[-1].time.isoformat() != pin["last_utc"]
            or any(b.time - a.time != timedelta(minutes=5) for a, b in zip(bars, bars[1:]))):
        raise LaunchDataFailure("daily bar timestamp pin changed")
    return bars


def _funding_rows(source: dict) -> list[Funding]:
    if source["source"] == "checksum_zip":
        url = source["url"]
        body = _verified_zip(url, source["sha256"], source["checksum_sha256"], 500_000)
        rows = _csv_rows(body, url.rsplit("/", 1)[-1], 2_000_000)
        if not rows or rows.pop(0) != ["calc_time", "funding_interval_hours", "last_funding_rate"]:
            raise LaunchDataFailure("funding CSV header changed")
        if len(rows) != source["rows"]:
            raise LaunchDataFailure("funding CSV row count changed")
        try:
            events = [Funding(datetime.fromtimestamp(int(r[0]) / 1000, timezone.utc), float(r[2]))
                      for r in rows]
        except (ValueError, IndexError) as exc:
            raise LaunchDataFailure("invalid funding CSV row") from exc
    elif source["source"] == "historical_rest_gap_fallback":
        try:
            response = requests.get(REST, params=source["request_params"], timeout=(10, 40))
        except requests.RequestException as exc:
            raise LaunchDataFailure("REST funding transport failure") from exc
        if response.status_code != 200 or sha256(response.content).hexdigest() != source["response_sha256"]:
            raise LaunchDataFailure("historical REST funding response changed")
        rows = response.json()
        if not isinstance(rows, list) or len(rows) != source["rows"]:
            raise LaunchDataFailure("REST funding row count changed")
        try:
            events = [Funding(datetime.fromtimestamp(int(r["fundingTime"]) / 1000, timezone.utc),
                              float(r["fundingRate"])) for r in rows]
        except (ValueError, KeyError, TypeError) as exc:
            raise LaunchDataFailure("invalid REST funding row") from exc
    else:
        raise LaunchDataFailure("unexpected funding source")
    if (not events or not all(math.isfinite(e.rate) and abs(e.rate) <= 1 for e in events)
            or any(b.time <= a.time for a, b in zip(events, events[1:]))):
        raise LaunchDataFailure("funding values/times invalid")
    return events


def _event(bar_pin: dict, funding_pin: dict) -> EventTape:
    if (bar_pin["article_code"] != funding_pin["article_code"]
            or bar_pin["contract_symbol"] != funding_pin["contract_symbol"]):
        raise LaunchDataFailure("bar/funding event identity mismatch")
    scheduled = datetime.fromisoformat(bar_pin["launch_utc"])
    effective = datetime.fromisoformat(funding_pin["effective_launch_utc"])
    entry, exit_time = (datetime.fromisoformat(funding_pin[k])
                        for k in ("entry_utc", "exit_utc"))
    first = _bar_rows(bar_pin["launch_day"])
    second = _bar_rows(bar_pin["next_day"])
    if effective != max(scheduled, first[0].time):
        raise LaunchDataFailure("effective launch time changed")
    bars = {bar.time: bar for bar in first + second}
    window = tuple(bars.get(effective + timedelta(minutes=5 * i))
                   for i in range(int((exit_time - effective) / timedelta(minutes=5)) + 1))
    if (entry != effective + timedelta(minutes=30) or exit_time != effective + timedelta(hours=24)
            or any(bar is None for bar in window)):
        raise LaunchDataFailure("required event bar grid changed")
    all_funding = [event for source in funding_pin["sources"] for event in _funding_rows(source)]
    held = tuple(sorted((event for event in all_funding if entry <= event.time < exit_time),
                        key=lambda event: event.time))
    if len(held) != funding_pin["funding_events_during_hold"]:
        raise LaunchDataFailure("funding event count changed")
    return EventTape(bar_pin["article_code"], bar_pin["contract_symbol"], scheduled,
                     effective, entry, exit_time, window, held)


def _btc_month(month: str, pin: str, expected_rows: int) -> dict[date, float]:
    name = f"BTCUSDT-5m-{month}.zip"
    url = f"{BTC_ROOT}/{name}"
    body = _verified_zip(url, pin, None, 8_000_000)
    rows = _csv_rows(body, name, 30_000_000)
    if rows and rows[0][0] == "open_time":
        rows.pop(0)
    if len(rows) != expected_rows:
        raise LaunchDataFailure(f"BTC month bar count changed: {month}")
    result = {}
    prior = None
    for row in rows:
        if len(row) != 12:
            raise LaunchDataFailure("BTC row columns changed")
        try:
            opened = int(row[0]); close = float(row[4])
        except ValueError as exc:
            raise LaunchDataFailure("invalid BTC source row") from exc
        if prior is not None and opened - prior != 300_000:
            raise LaunchDataFailure("BTC monthly grid gap")
        prior = opened
        if not math.isfinite(close) or close <= 0:
            raise LaunchDataFailure("invalid BTC daily close")
        at = datetime.fromtimestamp(opened / 1000, timezone.utc)
        if at.strftime("%Y-%m") != month:
            raise LaunchDataFailure("BTC row outside pinned month")
        if at.hour == 23 and at.minute == 55:
            result[at.date()] = close
    return result


def load_tape() -> LaunchTape:
    bar_bytes, funding_bytes = BAR_MANIFEST.read_bytes(), FUNDING_MANIFEST.read_bytes()
    bars, funding = json.loads(bar_bytes), json.loads(funding_bytes)
    if funding["bar_integrity_sha256"] != sha256(bar_bytes).hexdigest():
        raise LaunchDataFailure("funding manifest does not match bar manifest")
    lookup = {row["article_code"]: row for row in funding["rows"]}
    if len(bars["rows"]) != 215 or len(lookup) != 215:
        raise LaunchDataFailure("registered event count changed")
    events = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_event, row, lookup[row["article_code"]]): row
                   for row in bars["rows"]}
        for future in as_completed(futures):
            try:
                events.append(future.result())
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
    events.sort(key=lambda e: (e.effective, e.symbol))
    btc_pins = {}
    with BTC_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["symbol"] == "BTCUSDT" and "2020-12" <= row["month"] <= "2025-12":
                btc_pins[row["month"]] = (row["sha256"], int(row["bars"]))
    if len(btc_pins) != 61:
        raise LaunchDataFailure("BTC 61-month pin set changed")
    btc_close = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_btc_month, month, pin, count): month
                   for month, (pin, count) in btc_pins.items()}
        for future in as_completed(futures):
            try:
                btc_close.update(future.result())
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
    if len(btc_close) != 31 + 365 + 365 + 365 + 366 + 365:
        raise LaunchDataFailure("BTC completed daily close grid changed")
    return LaunchTape(tuple(events), btc_close,
                      {"bar_manifest_sha256": sha256(bar_bytes).hexdigest(),
                       "funding_manifest_sha256": sha256(funding_bytes).hexdigest(),
                       "btc_manifest_sha256": sha256(BTC_MANIFEST.read_bytes()).hexdigest()})
