"""Rate-blind 2026 funding archive audit for data-qualified 2020-vintage coins."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

from .basis_hourly_preflight import PreflightError, _fetch
from .crypto_2026_source_preflight import MONTHS, SYMBOLS


REPORT = Path(__file__).parent / "results/2026-11-07_funding_2026_source_audit.json"
MANIFEST = Path(__file__).parent / "preregistrations/2026-11-07_2026_funding_source_manifest.csv"
FIELDS = ("month", "market", "symbol", "dataset", "sha256", "events", "non_hour_events")


def audit_one(symbol: str, month: str) -> dict[str, str | int]:
    if symbol not in SYMBOLS or month not in MONTHS:
        raise ValueError("symbol/month outside data-only extension")
    name = f"{symbol}-fundingRate-{month}.zip"
    url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{name}"
    statement = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise PreflightError(f"official funding checksum malformed: {symbol} {month}")
    body = _fetch(url, limit=500_000)
    digest = sha256(body).hexdigest()
    if digest != match.group(1).lower():
        raise PreflightError(f"official funding ZIP hash mismatch: {symbol} {month}")
    year, number = map(int, month.split("-"))
    first = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp() * 1000)
    last = int(datetime(year + (number == 12), number % 12 + 1, 1,
                        tzinfo=timezone.utc).timestamp() * 1000)
    count, prior, non_hour = 0, -1, 0
    try:
        with ZipFile(io.BytesIO(body)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].flag_bits & 1 or members[0].file_size > 1_000_000):
                raise PreflightError(f"unexpected funding ZIP member: {symbol} {month}")
            with archive.open(members[0]) as raw:
                rows = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                if tuple(next(rows, ())) != ("calc_time", "funding_interval_hours", "last_funding_rate"):
                    raise PreflightError(f"funding schema changed: {symbol} {month}")
                for fields in rows:
                    if len(fields) != 3:
                        raise PreflightError(f"funding row shape changed: {symbol} {month}")
                    try:
                        stamp, interval = int(fields[0]), float(fields[1])
                    except ValueError as exc:
                        raise PreflightError(f"invalid funding time or interval: {symbol} {month}") from exc
                    if (not first <= stamp < last or stamp <= prior
                            or interval not in (1.0, 2.0, 4.0, 8.0)):
                        raise PreflightError(f"invalid funding time or interval: {symbol} {month}")
                    non_hour += int(stamp % 3_600_000 >= 1_000)
                    prior = stamp
                    count += 1
    except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
        raise PreflightError(f"invalid funding ZIP/CSV: {symbol} {month}") from exc
    if count <= 0 or non_hour:
        raise PreflightError(f"missing/off-hour funding event: {symbol} {month}")
    return {"month": month, "market": "um", "symbol": symbol,
            "dataset": "fundingRate", "sha256": digest, "events": count,
            "non_hour_events": non_hour}


def main() -> None:
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(audit_one, symbol, month): (symbol, month)
                for symbol in SYMBOLS for month in MONTHS}
        for job in as_completed(jobs):
            symbol, month = jobs[job]
            try:
                rows.append(job.result())
            except (PreflightError, UnicodeError, ValueError) as exc:
                errors.append(f"{symbol} {month}: {exc}")
    rows.sort(key=lambda row: (row["month"], row["symbol"]))
    REPORT.write_text(json.dumps({
        "kind": "source_funding_time_only", "period": "2026-01 through 2026-10",
        "rates_read": False, "strategy_results_read": False,
        "archives_checked": len(rows), "errors": sorted(errors),
        "complete_symbols": [symbol for symbol in SYMBOLS if sum(
            row["symbol"] == symbol for row in rows) == len(MONTHS)],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"2026 funding time audit: {len(rows)}/{len(SYMBOLS) * len(MONTHS)} archives, "
          f"{len(errors)} errors", flush=True)


if __name__ == "__main__":
    main()
