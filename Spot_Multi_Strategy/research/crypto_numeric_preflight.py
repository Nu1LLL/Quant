"""Price-output-blind numeric audit of checksum-pinned Binance 5m archives.

This is a reusable source-data quality check, not a strategy runner. It reads
numeric source columns only to validate individual bar invariants; it emits
no prices, returns, signals, rankings or PnL and retains no ZIP payloads.
"""

from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

from .basis_hourly_preflight import PreflightError, _fetch


MANIFEST = Path(__file__).parent / "preregistrations/2026-11-07_leadlag_5m_timestamp_manifest.csv"
OUTPUT = Path(__file__).parent / "results/2026-11-07_5m_numeric_source_audit.json"
FIELDS = ("month", "market", "symbol", "dataset", "sha256", "bars")
FIRST = (2020, 12)
LAST = (2025, 12)


def _months() -> tuple[str, ...]:
    return tuple(f"{year}-{month:02d}"
                 for year in range(FIRST[0], LAST[0] + 1)
                 for month in range(1, 13)
                 if FIRST <= (year, month) <= LAST)


def _manifest() -> dict[str, list[dict[str, str]]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise PreflightError("source manifest schema changed")
        rows = list(reader)
    by_symbol: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        symbol, month = row["symbol"], row["month"]
        if (row["market"] != "um" or row["dataset"] != "5m"
                or not re.fullmatch(r"[A-Z0-9]+USDT", symbol)
                or month not in _months()
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise PreflightError("source manifest row changed")
        year, number = map(int, month.split("-"))
        if row["bars"] != str(monthrange(year, number)[1] * 288):
            raise PreflightError("source manifest bar count changed")
        by_symbol.setdefault(symbol, []).append(row)
    expected = _months()
    if not by_symbol or any(tuple(row["month"] for row in values) != expected
                            for values in by_symbol.values()):
        raise PreflightError("source manifest months incomplete, duplicated or reordered")
    return by_symbol


def _row_flags(fields: list[str]) -> tuple[str, ...]:
    """Return source-invariant names only; never return numeric magnitudes."""
    if len(fields) != 12:
        return ("row_shape",)
    try:
        o, h, l, c, volume, quote, taker_base, taker_quote, ignored = (
            float(fields[index]) for index in (1, 2, 3, 4, 5, 7, 9, 10, 11))
        trades = int(fields[8])
    except (ValueError, OverflowError):
        return ("numeric_parse",)
    if not all(math.isfinite(value) for value in (
            o, h, l, c, volume, quote, taker_base, taker_quote, ignored)):
        return ("nonfinite",)
    flags = []
    if min(o, h, l, c) <= 0 or h < max(o, c, l) or l > min(o, c, h):
        flags.append("ohlc_bounds")
    if min(volume, quote, taker_base, taker_quote) < 0:
        flags.append("negative_volume")
    if taker_base > volume + 1e-9:
        flags.append("taker_base_exceeds_total")
    if taker_quote > quote + 1e-6:
        flags.append("taker_quote_exceeds_total")
    if trades < 0:
        flags.append("negative_trade_count")
    return tuple(flags)


def audit_archive(row: dict[str, str]) -> dict[str, object]:
    symbol, month = row["symbol"], row["month"]
    name = f"{symbol}-5m-{month}.zip"
    url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/5m/{name}"
    statement = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise PreflightError(f"official checksum malformed: {symbol} {month}")
    body = _fetch(url, limit=5_000_000)
    digest = sha256(body).hexdigest()
    if digest != row["sha256"] or digest != match.group(1).lower():
        raise PreflightError(f"pinned/official ZIP hash mismatch: {symbol} {month}")
    year, number = map(int, month.split("-"))
    expected = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp() * 1000)
    required = monthrange(year, number)[1] * 288
    counts: dict[str, int] = {}
    first_at: dict[str, str] = {}
    count = 0
    try:
        with ZipFile(io.BytesIO(body)) as archive:
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
                        counts[flag] = counts.get(flag, 0) + 1
                        first_at.setdefault(flag, datetime.fromtimestamp(
                            expected / 1000, tz=timezone.utc).isoformat())
                    count += 1
                    expected += 300_000
    except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
        raise PreflightError(f"invalid ZIP or CSV: {symbol} {month}") from exc
    if count != required or count != int(row["bars"]):
        raise PreflightError(f"five-minute row count changed: {symbol} {month}")
    return {"month": month, "symbol": symbol, "bars_checked": count,
            "issue_counts": counts, "first_issue_utc": first_at}


def run_audit() -> dict[str, object]:
    manifest = _manifest()
    # A source anomaly is reported, never repaired. Audit remaining months so
    # a future independent hypothesis can choose its universe before results.
    def per_symbol(symbol: str, rows: list[dict[str, str]]) -> tuple[str, list[dict[str, object]], list[str]]:
        valid = []
        errors = []
        for row in rows:
            try:
                valid.append(audit_archive(row))
            except (PreflightError, UnicodeError, ValueError) as exc:
                errors.append(f"{row['month']}: {exc}")
        return symbol, valid, errors

    reports: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=min(len(manifest), 10)) as pool:
        jobs = [pool.submit(per_symbol, symbol, rows)
                for symbol, rows in sorted(manifest.items())]
        for job in as_completed(jobs):
            symbol, months, errors = job.result()
            issues = [month for month in months if month["issue_counts"]]
            reports[symbol] = {
                "status": "pass" if len(months) == len(_months()) and not issues and not errors else "reject",
                "months_checked": len(months),
                "bars_checked": sum(int(month["bars_checked"]) for month in months),
                "numeric_issue_months": issues,
                "source_errors": errors,
            }
            print(f"numeric source audit: {symbol} {reports[symbol]['status']}", flush=True)
    return {"kind": "source_data_quality_only", "market": "Binance USD-M 5m",
            "period": "2020-12 through 2025-12", "manifest": str(MANIFEST.name),
            "strategy_results_read": False, "prices_or_returns_output": False,
            "symbols": {symbol: reports[symbol] for symbol in sorted(reports)}}


def main() -> None:
    report = run_audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"numeric source audit saved: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
