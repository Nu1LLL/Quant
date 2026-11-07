"""Data-only extension of full-grid Binance USD-M 5m audit through 2026-10.

The candidate coins were determined by the 2020-12 capitalization snapshot and
the 2020-12--2025-12 numeric source audit, before this period is opened.
Only source integrity flags and archive pins are persisted.
"""

from __future__ import annotations

from calendar import monthrange
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
from .crypto_numeric_preflight import _row_flags


SYMBOLS = ("ADAUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "ETHUSDT",
           "LINKUSDT", "THETAUSDT", "XMRUSDT")
MONTHS = tuple(f"2026-{month:02d}" for month in range(1, 11))
REPORT = Path(__file__).parent / "results/2026-11-07_5m_2026_source_audit.json"
MANIFEST = Path(__file__).parent / "preregistrations/2026-11-07_2026_5m_source_manifest.csv"
FIELDS = ("month", "market", "symbol", "dataset", "sha256", "bars")


def audit_one(symbol: str, month: str) -> tuple[dict[str, str | int], dict[str, object]]:
    if symbol not in SYMBOLS or month not in MONTHS:
        raise ValueError("symbol/month outside data-only extension")
    name = f"{symbol}-5m-{month}.zip"
    url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/5m/{name}"
    statement = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise PreflightError(f"official checksum malformed: {symbol} {month}")
    payload = _fetch(url, limit=5_000_000)
    digest = sha256(payload).hexdigest()
    if digest != match.group(1).lower():
        raise PreflightError(f"official ZIP hash mismatch: {symbol} {month}")
    year, number = map(int, month.split("-"))
    required = monthrange(year, number)[1] * 288
    expected = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp() * 1000)
    issues: dict[str, int] = {}
    first_at: dict[str, str] = {}
    count = 0
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].flag_bits & 1 or members[0].file_size > 10_000_000):
                raise PreflightError(f"unexpected ZIP member: {symbol} {month}")
            with archive.open(members[0]) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""),
                                    strict=True)
                for fields in reader:
                    if count == 0 and fields and fields[0] == "open_time":
                        continue
                    if (count >= required or len(fields) != 12
                            or fields[0] != str(expected)
                            or fields[6] != str(expected + 300_000 - 1)):
                        raise PreflightError(f"five-minute grid changed: {symbol} {month}, row {count}")
                    for flag in _row_flags(fields):
                        issues[flag] = issues.get(flag, 0) + 1
                        first_at.setdefault(flag, datetime.fromtimestamp(
                            expected / 1000, tz=timezone.utc).isoformat())
                    count += 1
                    expected += 300_000
    except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
        raise PreflightError(f"invalid ZIP/CSV: {symbol} {month}") from exc
    if count != required:
        raise PreflightError(f"five-minute row count changed: {symbol} {month}")
    pin = {"month": month, "market": "um", "symbol": symbol,
           "dataset": "5m", "sha256": digest, "bars": count}
    status = {"month": month, "symbol": symbol, "bars_checked": count,
              "numeric_issue_counts": issues, "first_issue_utc": first_at}
    return pin, status


def main() -> None:
    pins = []
    reports = []
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(audit_one, symbol, month): (symbol, month)
                for symbol in SYMBOLS for month in MONTHS}
        for job in as_completed(jobs):
            symbol, month = jobs[job]
            try:
                pin, report = job.result()
                pins.append(pin)
                reports.append(report)
            except (PreflightError, UnicodeError, ValueError) as exc:
                errors.append(f"{symbol} {month}: {exc}")
    pins.sort(key=lambda row: (row["month"], row["symbol"]))
    reports.sort(key=lambda row: (row["month"], row["symbol"]))
    output = {
        "kind": "source_data_quality_only", "market": "Binance USD-M 5m",
        "period": "2026-01 through 2026-10", "strategy_results_read": False,
        "prices_or_returns_output": False, "archives_checked": len(pins),
        "issues": [row for row in reports if row["numeric_issue_counts"]],
        "errors": sorted(errors),
        "complete_symbols": [symbol for symbol in SYMBOLS if not any(
            row["symbol"] == symbol for row in reports if row["numeric_issue_counts"])
            and not any(error.startswith(symbol + " ") for error in errors)
            and sum(row["symbol"] == symbol for row in reports) == len(MONTHS)],
    }
    REPORT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(pins)
    print(f"2026 source audit: {len(pins)}/{len(SYMBOLS) * len(MONTHS)} archives, "
          f"{len(output['complete_symbols'])} complete symbols, "
          f"{len(output['issues'])} issue months, {len(errors)} errors", flush=True)


if __name__ == "__main__":
    main()
