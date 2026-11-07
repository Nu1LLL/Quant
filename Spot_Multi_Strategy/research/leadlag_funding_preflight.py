"""Rate-blind funding-time audit for the ten-coin lead-lag candidate.

Only ZIP hashes, schema, contract identity and event timestamps/intervals are
inspected; rate values are never accessed by this data-only preflight.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

from .basis_hourly_preflight import PreflightError, _fetch
from .leadlag_5m_preflight import SYMBOLS


FIRST = (2021, 1)
LAST = (2025, 12)
OUTPUT = Path(__file__).parent / "preregistrations/2026-11-07_leadlag_funding_timestamp_manifest.csv"
FIELDS = ("month", "market", "symbol", "dataset", "sha256", "events", "non_hour_events")


def months() -> tuple[str, ...]:
    return tuple(f"{year}-{month:02d}"
                 for year in range(FIRST[0], LAST[0] + 1)
                 for month in range(1, 13)
                 if FIRST <= (year, month) <= LAST)


def audit_one(symbol: str, month: str) -> dict[str, str | int]:
    if symbol not in SYMBOLS or month not in months():
        raise ValueError("symbol or month outside data-only plan")
    name = f"{symbol}-fundingRate-{month}.zip"
    url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{name}"
    statement = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise PreflightError(f"malformed funding checksum: {symbol} {month}")
    body = _fetch(url, limit=500_000)
    digest = sha256(body).hexdigest()
    if digest != match.group(1).lower():
        raise PreflightError(f"funding ZIP/checksum mismatch: {symbol} {month}")
    year, number = map(int, month.split("-"))
    first = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp() * 1000)
    last = (int(datetime(year + (number == 12), number % 12 + 1, 1,
                         tzinfo=timezone.utc).timestamp() * 1000))
    count = 0
    prior = -1
    non_hour = 0
    try:
        with ZipFile(io.BytesIO(body)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].file_size > 1_000_000 or members[0].flag_bits & 1):
                raise PreflightError(f"unexpected funding ZIP member: {symbol} {month}")
            with archive.open(members[0]) as raw:
                rows = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                if tuple(next(rows, ())) != ("calc_time", "funding_interval_hours", "last_funding_rate"):
                    raise PreflightError(f"funding schema changed: {symbol} {month}")
                for fields in rows:
                    if len(fields) != 3:
                        raise PreflightError(f"funding row shape changed: {symbol} {month}")
                    try:
                        stamp = int(fields[0])
                        interval = float(fields[1])
                    except ValueError as exc:
                        raise PreflightError(f"invalid funding timestamp or interval: {symbol} {month}") from exc
                    if (not first <= stamp < last or stamp <= prior
                            or interval not in (1.0, 2.0, 4.0, 8.0)):
                        raise PreflightError(f"invalid funding timestamp or interval: {symbol} {month}")
                    non_hour += int(stamp % 3_600_000 >= 1_000)
                    prior = stamp
                    count += 1
    except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
        raise PreflightError(f"bad funding ZIP or timestamp CSV: {symbol} {month}") from exc
    if not count or non_hour:
        raise PreflightError(f"missing or off-hour funding timestamp: {symbol} {month}, events={count}, off_hour={non_hour}")
    return {"month": month, "market": "um", "symbol": symbol,
            "dataset": "fundingRate", "sha256": digest,
            "events": count, "non_hour_events": non_hour}


def audit_all() -> tuple[list[dict[str, str | int]], dict[str, str]]:
    def symbol_run(symbol: str) -> tuple[str, list[dict[str, str | int]], str | None]:
        found = []
        for month in months():
            try:
                found.append(audit_one(symbol, month))
            except PreflightError as exc:
                return symbol, found, str(exc)
        return symbol, found, None

    complete: list[dict[str, str | int]] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as pool:
        futures = [pool.submit(symbol_run, symbol) for symbol in SYMBOLS]
        for future in as_completed(futures):
            symbol, rows, error = future.result()
            if error is None:
                complete.extend(rows)
                print(f"funding time-only: {symbol} full period passed", flush=True)
            else:
                failures[symbol] = error
                print(f"funding time-only: {symbol} rejected at {len(rows)}/{len(months())} months: {error}", flush=True)
    complete.sort(key=lambda row: (row["month"], SYMBOLS.index(str(row["symbol"]))))
    return complete, failures


def main() -> None:
    rows, failures = audit_all()
    if failures:
        print(f"funding time-only audit rejected: {len(failures)} symbols; no strategy is registered", flush=True)
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"funding time-only manifest: {OUTPUT}, {len(rows)} ZIP pins", flush=True)


if __name__ == "__main__":
    main()
