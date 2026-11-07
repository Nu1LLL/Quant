"""Prospective, read-only-market-data journal for BTC exchange supply.

Each successful poll records its response hash and only changed daily values.
An as-of reader can never see a value before this collector first observed it.
This is not a trading signal, a historical PIT backtest or Coin Metrics' paid
``SplyExPitNtv`` methodology. The local journal is non-commercial research.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
import argparse
import json
import re
import sqlite3
import time

import requests


BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
ASSET = "btc"
METRIC = "SplyExNtv"
LOOKBACK_DAYS = 63
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T00:00:00(?:\.0{1,9})?Z\Z")
_ISO_UTC = "%Y-%m-%dT%H:%M:%S.%fZ"
MAX_BODY = 200_000


class JournalError(RuntimeError):
    """Source, schema, clock or local append-only journal failure."""


@dataclass(frozen=True)
class PollResult:
    fetched_at: datetime
    rows: int
    new_versions: int
    response_sha256: str


@dataclass(frozen=True)
class AsOfValue:
    observed_date: date
    first_seen_at: datetime
    value: Decimal


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise JournalError("UTC-aware datetime required")
    result = value.astimezone(timezone.utc)
    if result.year < 2020 or result.year > 2100:
        raise JournalError("clock outside supported years")
    return result


def _query(poll_time: datetime) -> tuple[str, date, date]:
    now = _utc(poll_time)
    start = now.date() - timedelta(days=LOOKBACK_DAYS)
    end = now.date() - timedelta(days=1)
    params = {
        "assets": ASSET, "metrics": METRIC, "frequency": "1d",
        "start_time": start.isoformat(), "end_time": end.isoformat(),
        "page_size": "100", "paging_from": "start",
    }
    return BASE + "?" + urlencode(params), start, end


def _download(url: str) -> bytes:
    for attempt in range(4):
        try:
            with requests.get(url, timeout=(10, 30), stream=True) as response:
                if response.status_code == 200:
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=32_768):
                        body.extend(chunk)
                        if len(body) > MAX_BODY:
                            raise JournalError("Coin Metrics response exceeds bound")
                    if body:
                        return bytes(body)
                status = response.status_code
        except JournalError:
            raise
        except requests.RequestException as exc:
            if attempt == 3:
                raise JournalError("Coin Metrics request failed") from exc
        else:
            if status != 429 and not 500 <= status <= 599:
                raise JournalError(f"Coin Metrics HTTP {status} or empty body")
            if attempt == 3:
                raise JournalError(f"Coin Metrics HTTP {status}")
        time.sleep((0.5, 1.0, 2.0)[attempt])
    raise AssertionError("unreachable")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise JournalError("duplicate Coin Metrics JSON key")
        result[key] = value
    return result


def _parse(body: bytes, start: date, end: date) -> tuple[tuple[date, str], ...]:
    if not isinstance(body, bytes) or not body or len(body) > MAX_BODY:
        raise JournalError("empty or oversized Coin Metrics response")
    try:
        payload = json.loads(body.decode("utf-8-sig"), object_pairs_hook=_unique_pairs)
    except (ValueError, UnicodeError) as exc:
        raise JournalError("invalid Coin Metrics JSON") from exc
    if not isinstance(payload, dict) or payload.get("next_page_url") is not None:
        raise JournalError("unexpected Coin Metrics pagination")
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows or len(rows) > 100:
        raise JournalError("invalid Coin Metrics row count")
    output = []
    previous = None
    for row in rows:
        if not isinstance(row, dict) or row.get("asset") != ASSET or METRIC not in row:
            raise JournalError("wrong asset or missing exchange supply")
        raw_time = row.get("time")
        if not isinstance(raw_time, str) or not _STAMP.fullmatch(raw_time):
            raise JournalError("daily observation is not UTC midnight")
        try:
            observed = date.fromisoformat(raw_time[:10])
        except ValueError as exc:
            raise JournalError("invalid observation date") from exc
        if not start <= observed <= end or (previous is not None and observed <= previous):
            raise JournalError("observation outside query window or out of order")
        raw = row[METRIC]
        if not isinstance(raw, str) or not raw or len(raw) > 128 or raw.strip() != raw:
            raise JournalError("exchange supply must be a decimal string")
        try:
            value = Decimal(raw)
        except InvalidOperation as exc:
            raise JournalError("invalid exchange supply decimal") from exc
        if not value.is_finite() or value < 0 or abs(value.as_tuple().exponent) > 64:
            raise JournalError("nonfinite or negative exchange supply")
        output.append((observed, raw))
        previous = observed
    return tuple(output)


def _schema(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS polls (
            id INTEGER PRIMARY KEY,
            fetched_at_utc TEXT NOT NULL UNIQUE,
            source_url TEXT NOT NULL,
            response_sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            first_observed_date TEXT NOT NULL,
            last_observed_date TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS versions (
            id INTEGER PRIMARY KEY,
            observed_date TEXT NOT NULL,
            first_seen_at_utc TEXT NOT NULL,
            value_text TEXT NOT NULL,
            source_poll_id INTEGER NOT NULL REFERENCES polls(id)
        );
        CREATE INDEX IF NOT EXISTS versions_asof
            ON versions(observed_date, first_seen_at_utc DESC, id DESC);
    """)


def capture_current(path: str | Path, *, fetcher: Callable[[str], bytes] | None = None,
                    poll_time: datetime | None = None,
                    received_at: datetime | None = None) -> PollResult:
    """Append one source poll; injected clock/fetcher support offline tests."""
    started = _utc(poll_time or datetime.now(timezone.utc))
    url, start, end = _query(started)
    body = (fetcher or _download)(url)
    rows = _parse(body, start, end)
    received = _utc(received_at or datetime.now(timezone.utc))
    if received < started:
        raise JournalError("response received before query start")
    stamp = received.strftime(_ISO_UTC)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(target, timeout=15) as db:
            db.execute("PRAGMA foreign_keys=ON")
            _schema(db)
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT MAX(fetched_at_utc) FROM polls").fetchone()[0]
            if prior is not None and stamp <= prior:
                raise JournalError("poll clock did not advance monotonically")
            cursor = db.execute(
                "INSERT INTO polls(fetched_at_utc,source_url,response_sha256,row_count,first_observed_date,last_observed_date) VALUES (?,?,?,?,?,?)",
                (stamp, url, sha256(body).hexdigest(), len(rows),
                 rows[0][0].isoformat(), rows[-1][0].isoformat()),
            )
            poll_id = cursor.lastrowid
            changed = 0
            for observed, value in rows:
                previous = db.execute(
                    "SELECT value_text FROM versions WHERE observed_date=? ORDER BY id DESC LIMIT 1",
                    (observed.isoformat(),),
                ).fetchone()
                if previous is None or Decimal(previous[0]) != Decimal(value):
                    db.execute(
                        "INSERT INTO versions(observed_date,first_seen_at_utc,value_text,source_poll_id) VALUES (?,?,?,?)",
                        (observed.isoformat(), stamp, value, poll_id),
                    )
                    changed += 1
            db.commit()
    except sqlite3.Error as exc:
        raise JournalError("local exchange-supply journal failure") from exc
    return PollResult(received, len(rows), changed, sha256(body).hexdigest())


def read_asof(path: str | Path, observed: date, decision_time: datetime) -> AsOfValue | None:
    """Read only a version actually observed by the specified decision time."""
    decision = _utc(decision_time)
    if type(observed) is not date or observed >= decision.date():
        raise JournalError("observed date must precede decision UTC date")
    target = Path(path)
    if not target.is_file():
        return None
    uri = f"file:{target.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=15) as db:
            row = db.execute(
                "SELECT first_seen_at_utc,value_text FROM versions "
                "WHERE observed_date=? AND first_seen_at_utc<=? "
                "ORDER BY first_seen_at_utc DESC,id DESC LIMIT 1",
                (observed.isoformat(), decision.strftime(_ISO_UTC)),
            ).fetchone()
    except sqlite3.Error as exc:
        raise JournalError("local as-of journal read failed") from exc
    if row is None:
        return None
    first_seen = datetime.strptime(row[0], _ISO_UTC).replace(tzinfo=timezone.utc)
    return AsOfValue(observed, first_seen, Decimal(row[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture BTC exchange supply for non-trading prospective research")
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    result = capture_current(args.journal)
    print(f"exchange-supply poll: rows={result.rows}; new_versions={result.new_versions}; received_utc={result.fetched_at.isoformat()}")


if __name__ == "__main__":
    main()
