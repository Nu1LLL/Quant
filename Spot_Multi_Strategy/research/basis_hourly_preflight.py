"""Timestamp-only Binance spot/perp hourly archive preflight.

This tool is deliberately rate- and price-blind. It verifies official SHA,
ZIP member identity, and every hourly open/close timestamp before a possible
spot-perpetual basis pre-registration. It stores no original archive bytes.
"""

from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile


SYMBOLS = ("ADAUSDT", "BNBUSDT", "DOGEUSDT")
MARKETS = (("spot", "spot"), ("um", "futures/um"))
FIRST = (2020, 12)
LAST = (2025, 12)
MAX_ZIP = 2_000_000
MAX_CHECKSUM = 200
OUTPUT = Path(__file__).parent / "preregistrations/2026-11-07_basis_hourly_timestamp_manifest.csv"


class PreflightError(ValueError):
    pass


def _fetch(url: str, *, limit: int) -> bytes:
    for attempt in range(2):
        process = subprocess.run(
            ("curl", "--fail", "--location", "--silent", "--show-error",
             "--max-time", "60", "--proto", "=https", "--proto-redir", "=https", url),
            capture_output=True,
        )
        if process.returncode == 0:
            if len(process.stdout) > limit:
                raise PreflightError(f"response exceeds preflight limit: {url}")
            return process.stdout
    raise PreflightError(f"transport unavailable after two tries: {url}")


def _months():
    return (f"{y}-{m:02d}" for y in range(FIRST[0], LAST[0] + 1)
            for m in range(1, 13) if FIRST <= (y, m) <= LAST)


def audit_one(market: str, prefix: str, symbol: str, month: str) -> dict[str, str | int]:
    name = f"{symbol}-1h-{month}.zip"
    url = f"https://data.binance.vision/data/{prefix}/monthly/klines/{symbol}/1h/{name}"
    checksum = _fetch(url + ".CHECKSUM", limit=MAX_CHECKSUM).decode("ascii").strip()
    matched = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", checksum)
    if matched is None or matched.group(2) != name:
        raise PreflightError(f"malformed official checksum: {market} {symbol} {month}")
    payload = _fetch(url, limit=MAX_ZIP)
    digest = sha256(payload).hexdigest()
    if digest != matched.group(1).lower():
        raise PreflightError(f"official hash mismatch: {market} {symbol} {month}")
    year, number = map(int, month.split("-"))
    days = monthrange(year, number)[1]
    unit = 1_000_000 if market == "spot" and month >= "2025-01" else 1_000
    expected = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp()) * unit
    step = 3600 * unit
    count = 0
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].file_size > 5_000_000 or members[0].flag_bits & 1):
                raise PreflightError(f"unexpected ZIP member: {market} {symbol} {month}")
            with archive.open(members[0]) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""),
                                    strict=True)
                for fields in reader:
                    if count == 0 and fields and fields[0] == "open_time":
                        continue
                    if (len(fields) != 12 or fields[0] != str(expected)
                            or fields[6] != str(expected + step - 1)):
                        raise PreflightError(f"hourly grid gap at row {count}: {market} {symbol} {month}")
                    count += 1
                    expected += step
    except (OSError, ValueError, UnicodeError, csv.Error) as exc:
        if isinstance(exc, PreflightError):
            raise
        raise PreflightError(f"invalid ZIP or timestamp CSV: {market} {symbol} {month}") from exc
    if count != days * 24:
        raise PreflightError(f"truncated hourly grid ({count}/{days * 24}): {market} {symbol} {month}")
    return {"month": month, "market": market, "symbol": symbol,
            "dataset": "1h", "sha256": digest, "hours": count}


def audit_all() -> list[dict[str, str | int]]:
    requests = [(market, prefix, symbol, month)
                for month in _months() for market, prefix in MARKETS
                for symbol in SYMBOLS]
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(audit_one, *request): request for request in requests}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except PreflightError as exc:
                failures.append(str(exc))
            completed = len(rows) + len(failures)
            if completed % 48 == 0:
                print(f"timestamp-only preflight: {completed}/{len(requests)}", flush=True)
    if failures:
        raise PreflightError("timestamp-only preflight rejected: " + " | ".join(sorted(failures)))
    return sorted(rows, key=lambda row: (row["month"], row["market"], row["symbol"]))


def main() -> None:
    rows = audit_all()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("month", "market", "symbol", "dataset", "sha256", "hours"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"complete preflight: {len(rows)} pinned archives; {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
