"""Rate-blind five-altcoin funding-carry archive preflight.

Verify only full hourly time grids, funding timestamps, ZIP identities and
official checksums. Neither prices nor funding rates nor PnL are inspected.
"""

from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import io
from pathlib import Path
import re
from zipfile import ZipFile

from .basis_hourly_preflight import PreflightError, _fetch, audit_one


SYMBOLS = ("ADAUSDT", "BNBUSDT", "DOGEUSDT")
START_MONTH = (2020, 12)
KLINE_END_MONTH = (2026, 1)  # January 1 00:00 final exit-open is in this ZIP.
FUNDING_END_MONTH = (2026, 1)  # January 1 00:00 final held funding event.
FUNDING_START = (2021, 1)
KLINE_OUTPUT = Path(__file__).parent / "preregistrations/2026-11-07_three_alt_funding_kline_1h_manifest.csv"
FUNDING_OUTPUT = Path(__file__).parent / "preregistrations/2026-11-07_three_alt_funding_rate_manifest.csv"


def _months(first: tuple[int, int], last: tuple[int, int]):
    return (f"{y}-{m:02d}" for y in range(first[0], last[0] + 1)
            for m in range(1, 13) if first <= (y, m) <= last)


def audit_funding(symbol: str, month: str) -> dict[str, str | int]:
    name = f"{symbol}-fundingRate-{month}.zip"
    url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{name}"
    statement = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise PreflightError(f"malformed funding checksum: {symbol} {month}")
    payload = _fetch(url, limit=200_000)
    digest = sha256(payload).hexdigest()
    if digest != match.group(1).lower():
        raise PreflightError(f"funding hash mismatch: {symbol} {month}")
    year, number = map(int, month.split("-"))
    expected = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp() * 1000)
    count = 0
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].file_size > 500_000 or members[0].flag_bits & 1):
                raise PreflightError(f"unexpected funding ZIP member: {symbol} {month}")
            with archive.open(members[0]) as raw:
                for line in raw:
                    if count == 0 and line.startswith(b"calc_time,"):
                        if line.strip() != b"calc_time,funding_interval_hours,last_funding_rate":
                            raise PreflightError(f"funding schema changed: {symbol} {month}")
                        continue
                    parts = line.strip().split(b",", 2)
                    if len(parts) != 3:
                        raise PreflightError(f"malformed funding CSV row: {symbol} {month}")
                    stamp = int(parts[0])
                    interval = Decimal(parts[1].decode("ascii"))
                    if (not expected <= stamp < expected + 1000
                            or not interval.is_finite() or interval <= 0):
                        raise PreflightError(f"funding timestamp gap at event {count}: {symbol} {month}")
                    expected += 8 * 3600 * 1000
                    count += 1
    except (OSError, ValueError, UnicodeError, InvalidOperation) as exc:
        if isinstance(exc, PreflightError):
            raise
        raise PreflightError(f"invalid funding timestamp ZIP: {symbol} {month}") from exc
    required = monthrange(year, number)[1] * 3
    if count != required:
        raise PreflightError(f"incomplete funding grid ({count}/{required}): {symbol} {month}")
    return {"month": month, "market": "um", "symbol": symbol,
            "dataset": "fundingRate", "sha256": digest, "events": count}


def audit_all() -> tuple[list[dict], list[dict]]:
    requests = [("kline", symbol, month) for month in _months(START_MONTH, KLINE_END_MONTH)
                for symbol in SYMBOLS]
    requests += [("funding", symbol, month) for month in _months(FUNDING_START, FUNDING_END_MONTH)
                 for symbol in SYMBOLS]
    klines, funding, failures = [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        tasks = {
            pool.submit(audit_one, "um", "futures/um", symbol, month)
            if kind == "kline" else pool.submit(audit_funding, symbol, month):
            (kind, symbol, month)
            for kind, symbol, month in requests
        }
        for task in as_completed(tasks):
            kind, _, _ = tasks[task]
            try:
                (klines if kind == "kline" else funding).append(task.result())
            except PreflightError as exc:
                failures.append(str(exc))
            completed = len(klines) + len(funding) + len(failures)
            if completed % 60 == 0:
                print(f"funding-carry timestamp preflight: {completed}/{len(requests)}", flush=True)
    if failures:
        raise PreflightError("funding-carry data preflight rejected: " + " | ".join(sorted(failures)))
    klines.sort(key=lambda row: (row["month"], row["symbol"]))
    funding.sort(key=lambda row: (row["month"], row["symbol"]))
    return klines, funding


def main() -> None:
    klines, funding = audit_all()
    for path, rows, fields in (
        (KLINE_OUTPUT, klines, ("month", "market", "symbol", "dataset", "sha256", "hours")),
        (FUNDING_OUTPUT, funding, ("month", "market", "symbol", "dataset", "sha256", "events")),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(f"complete: {len(klines)} hourly ZIPs, {len(funding)} funding ZIPs", flush=True)


if __name__ == "__main__":
    main()
