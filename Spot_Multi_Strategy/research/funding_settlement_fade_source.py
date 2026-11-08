"""Frozen two-coin five-minute futures archives and actual funding clocks."""

from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import csv
import math
import re

import numpy as np

from research.crypto_numeric_preflight import _row_flags
from research.frozen_source_transport import IntegrityFailure, TransportFailure, fetch_pinned


ROOT = Path(__file__).resolve().parent
SYMBOLS = ("BTCUSDT", "ETHUSDT")
PRICE_MANIFEST = ROOT / "preregistrations/2026-11-07_leadlag_5m_timestamp_manifest.csv"
FUNDING_MANIFEST = ROOT / "preregistrations/2026-11-07_leadlag_funding_timestamp_manifest.csv"
PRICE_MANIFEST_SHA = "335ea7b79634756ee0cf82428fc32d6c95fa3477470984d00b26ce013a19328a"
FUNDING_MANIFEST_SHA = "09492df23bc57c88d620c506b134fe690f55672156fc468e642a02903a6e1389"
PRICE_MONTHS = tuple(f"{year}-{month:02d}" for year in range(2020, 2026)
                     for month in range(1, 13) if (year, month) >= (2020, 12))
FUNDING_MONTHS = PRICE_MONTHS[1:]
BASE_MS = int(datetime(2020, 12, 1, tzinfo=timezone.utc).timestamp() * 1000)
EVAL_MS = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
BAR_MS = 300_000
DAY_BARS = 288
WARMUP_BARS = 31 * DAY_BARS
EVAL_DAYS = 1_826


class SourceFailure(ValueError):
    pass


@dataclass(frozen=True)
class Tape:
    prices: dict[str, np.ndarray]  # open, high, low, close, quote volume
    funding_index: dict[str, frozenset[int]]
    source_audit: dict


def _pins(path: Path, digest: str, fields: tuple[str, ...], months: tuple[str, ...],
          dataset: str) -> dict[tuple[str, str], dict[str, str]]:
    try:
        body = path.read_bytes()
        if sha256(body).hexdigest() != digest:
            raise SourceFailure(f"{dataset}: frozen manifest SHA changed")
        reader = csv.DictReader(body.decode("utf-8-sig").splitlines())
        if tuple(reader.fieldnames or ()) != fields:
            raise SourceFailure(f"{dataset}: frozen manifest schema changed")
        rows = list(reader)
        selected = {}
        for row in rows:
            if row["symbol"] not in SYMBOLS or row["month"] not in months:
                continue
            key = (row["month"], row["symbol"])
            if (key in selected or row["market"] != "um" or row["dataset"] != dataset
                    or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
                raise SourceFailure(f"{dataset}: malformed or repeated manifest key {key}")
            y, m = map(int, row["month"].split("-"))
            count = int(row["bars" if dataset == "5m" else "events"])
            if dataset == "5m":
                if count != monthrange(y, m)[1] * DAY_BARS:
                    raise SourceFailure(f"{key}: price bar count changed")
            elif (count != monthrange(y, m)[1] * 3
                  or row["non_hour_events"] != "0"):
                raise SourceFailure(f"{key}: funding event metadata changed")
            selected[key] = row
        if len(selected) != len(months) * len(SYMBOLS):
            raise SourceFailure(f"{dataset}: selected manifest month grid incomplete")
        return selected
    except SourceFailure:
        raise
    except (OSError, UnicodeError, csv.Error, ValueError, KeyError, TypeError) as exc:
        raise SourceFailure(f"{dataset}: invalid frozen manifest: {exc}") from exc


def _archive(symbol: str, month: str, dataset: str, pin: str) -> tuple[ZipFile, dict]:
    if dataset == "5m":
        filename = f"{symbol}-5m-{month}.zip"
        url = ("https://data.binance.vision/data/futures/um/monthly/klines/"
               f"{symbol}/5m/{filename}")
        limit, member_limit = 5_000_000, 10_000_000
    else:
        filename = f"{symbol}-fundingRate-{month}.zip"
        url = ("https://data.binance.vision/data/futures/um/monthly/fundingRate/"
               f"{symbol}/{filename}")
        limit, member_limit = 500_000, 1_000_000
    try:
        sidecar = fetch_pinned(url + ".CHECKSUM", max_bytes=500)
        checksum = sidecar.body.decode("ascii").strip()
        if (re.fullmatch(r"[0-9a-fA-F]{64}[ \t]+" + re.escape(filename),
                         checksum) is None or checksum.split()[0].lower() != pin):
            raise SourceFailure(f"{symbol} {month} {dataset}: official checksum changed")
        receipt = fetch_pinned(url, expected_sha256=pin, max_bytes=limit)
        archive = ZipFile(BytesIO(receipt.body))
        members = archive.infolist()
        if (len(members) != 1 or members[0].filename != filename[:-4] + ".csv"
                or members[0].file_size > member_limit or members[0].flag_bits & 1):
            archive.close()
            raise SourceFailure(f"{symbol} {month} {dataset}: ZIP member invalid")
        audit = {"symbol": symbol, "month": month, "dataset": dataset,
                 "sha256": receipt.sha256, "bytes": len(receipt.body),
                 "checksum_attempts": sidecar.attempts,
                 "zip_attempts": receipt.attempts}
        return archive, audit
    except SourceFailure:
        raise
    except (TransportFailure, IntegrityFailure, BadZipFile, UnicodeError) as exc:
        raise SourceFailure(f"{symbol} {month} {dataset}: frozen source failed: {exc}") from exc


def _price_month(symbol: str, month: str, pin: dict[str, str]) -> tuple[np.ndarray, dict]:
    y, m = map(int, month.split("-"))
    required = monthrange(y, m)[1] * DAY_BARS
    expected = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)
    values = np.empty((required, 5), dtype=np.float64)
    count = 0
    archive, audit = _archive(symbol, month, "5m", pin["sha256"])
    try:
        with archive:
            with archive.open(archive.infolist()[0]) as raw:
                reader = csv.reader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                for fields in reader:
                    if count == 0 and fields and fields[0] == "open_time":
                        continue
                    if (count >= required or len(fields) != 12 or
                            fields[0] != str(expected) or
                            fields[6] != str(expected + BAR_MS - 1)):
                        raise SourceFailure(f"{symbol} {month}: five-minute grid changed at {count}")
                    flags = _row_flags(fields)
                    if flags:
                        raise SourceFailure(f"{symbol} {month}: price invariant {flags} at {count}")
                    values[count] = tuple(float(fields[k]) for k in (1, 2, 3, 4, 7))
                    expected += BAR_MS
                    count += 1
    except (BadZipFile, OSError, UnicodeError, csv.Error, ValueError) as exc:
        if isinstance(exc, SourceFailure):
            raise
        raise SourceFailure(f"{symbol} {month}: invalid price CSV: {exc}") from exc
    if count != required or count != int(pin["bars"]):
        raise SourceFailure(f"{symbol} {month}: price month incomplete")
    return values, audit


def _funding_month(symbol: str, month: str, pin: dict[str, str]) -> tuple[tuple[int, ...], dict]:
    y, m = map(int, month.split("-"))
    first = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)
    last = int(datetime(y + (m == 12), m % 12 + 1, 1,
                        tzinfo=timezone.utc).timestamp() * 1000)
    stamps = []
    archive, audit = _archive(symbol, month, "fundingRate", pin["sha256"])
    try:
        with archive:
            with archive.open(archive.infolist()[0]) as raw:
                reader = csv.reader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                if tuple(next(reader, ())) != ("calc_time", "funding_interval_hours",
                                                "last_funding_rate"):
                    raise SourceFailure(f"{symbol} {month}: funding header changed")
                for fields in reader:
                    if len(fields) != 3:
                        raise SourceFailure(f"{symbol} {month}: funding row width changed")
                    stamp = int(fields[0])
                    interval, rate = float(fields[1]), float(fields[2])
                    if (not first <= stamp < last or stamp % 3_600_000 >= 1_000
                            or (stamp // 3_600_000) % 24 not in (0, 8, 16)
                            or interval != 8.0 or not math.isfinite(rate)
                            or abs(rate) > .1):
                        raise SourceFailure(f"{symbol} {month}: invalid funding event")
                    normalized = stamp - stamp % 3_600_000
                    if stamps and normalized - stamps[-1] != 8 * 3_600_000:
                        raise SourceFailure(f"{symbol} {month}: funding interval changed")
                    stamps.append(normalized)
    except (BadZipFile, OSError, UnicodeError, csv.Error, ValueError) as exc:
        if isinstance(exc, SourceFailure):
            raise
        raise SourceFailure(f"{symbol} {month}: invalid funding CSV: {exc}") from exc
    if (not stamps or len(stamps) != int(pin["events"]) or
            stamps[0] != first or stamps[-1] != last - 8 * 3_600_000):
        raise SourceFailure(f"{symbol} {month}: funding event grid incomplete")
    indices = tuple((stamp - BASE_MS) // BAR_MS for stamp in stamps)
    return indices, audit


def load_tape() -> Tape:
    price_pins = _pins(PRICE_MANIFEST, PRICE_MANIFEST_SHA,
                       ("month", "market", "symbol", "dataset", "sha256", "bars"),
                       PRICE_MONTHS, "5m")
    funding_pins = _pins(FUNDING_MANIFEST, FUNDING_MANIFEST_SHA,
                         ("month", "market", "symbol", "dataset", "sha256",
                          "events", "non_hour_events"), FUNDING_MONTHS, "fundingRate")
    price_parts = {symbol: [] for symbol in SYMBOLS}
    price_audit = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for month in PRICE_MONTHS:
            jobs = {symbol: pool.submit(_price_month, symbol, month,
                                        price_pins[(month, symbol)]) for symbol in SYMBOLS}
            for symbol in SYMBOLS:
                arr, audit = jobs[symbol].result()
                price_parts[symbol].append(arr)
                price_audit.append(audit)
    prices = {symbol: np.concatenate(price_parts[symbol]) for symbol in SYMBOLS}
    total = WARMUP_BARS + EVAL_DAYS * DAY_BARS
    if any(arr.shape != (total, 5) or not np.isfinite(arr).all()
           for arr in prices.values()):
        raise SourceFailure("five-minute tape length or finiteness changed")
    funding = {symbol: [] for symbol in SYMBOLS}
    funding_audit = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for month in FUNDING_MONTHS:
            jobs = {symbol: pool.submit(_funding_month, symbol, month,
                                        funding_pins[(month, symbol)]) for symbol in SYMBOLS}
            for symbol in SYMBOLS:
                indices, audit = jobs[symbol].result()
                funding[symbol].extend(indices)
                funding_audit.append(audit)
    expected = tuple(WARMUP_BARS + i * 96 for i in range(EVAL_DAYS * 3))
    for symbol in SYMBOLS:
        if tuple(funding[symbol]) != expected:
            raise SourceFailure(f"{symbol}: funding clock incomplete or differs across months")
        prices[symbol].setflags(write=False)
    return Tape(prices, {symbol: frozenset(funding[symbol]) for symbol in SYMBOLS}, {
        "price_manifest_sha256": PRICE_MANIFEST_SHA,
        "funding_manifest_sha256": FUNDING_MANIFEST_SHA,
        "price_archives": len(price_audit), "funding_archives": len(funding_audit),
        "funding_events_per_symbol": len(expected),
        "transport_retries": sum(a["checksum_attempts"] + a["zip_attempts"] - 2
                                 for a in price_audit + funding_audit),
        "archives": price_audit + funding_audit,
    })
