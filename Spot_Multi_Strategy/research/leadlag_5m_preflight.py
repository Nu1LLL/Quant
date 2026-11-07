"""Price-blind, checksum-pinned five-minute archive audit for lead-lag research.

The candidate list is the data-available subset of the largest non-stablecoin
assets in CoinMarketCap's 2020-12-30 snapshot. Earlier three-date probes
excluded XRP, LTC, XLM, TRX and XEM for a 2022-02 timestamp gap; BSV and EOS
had unavailable boundary archives. No price, volume, return or PnL is read.
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
from zipfile import BadZipFile, ZipFile

from .basis_hourly_preflight import PreflightError, _fetch


SYMBOLS = ("BTCUSDT", "ETHUSDT", "BCHUSDT", "DOTUSDT", "ADAUSDT",
           "BNBUSDT", "LINKUSDT", "XMRUSDT", "THETAUSDT", "XTZUSDT")
FIRST = (2020, 12)
LAST = (2025, 12)
OUTPUT = Path(__file__).parent / "preregistrations/2026-11-07_leadlag_5m_timestamp_manifest.csv"
FIELDS = ("month", "market", "symbol", "dataset", "sha256", "bars")


def months() -> tuple[str, ...]:
    return tuple(f"{year}-{month:02d}"
                 for year in range(FIRST[0], LAST[0] + 1)
                 for month in range(1, 13)
                 if FIRST <= (year, month) <= LAST)


def audit_one(symbol: str, month: str) -> dict[str, str | int]:
    if symbol not in SYMBOLS or month not in months():
        raise ValueError("symbol or month outside data-only plan")
    name = f"{symbol}-5m-{month}.zip"
    url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/5m/{name}"
    statement = _fetch(url + ".CHECKSUM", limit=200).decode("ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise PreflightError(f"malformed checksum: {symbol} {month}")
    body = _fetch(url, limit=5_000_000)
    digest = sha256(body).hexdigest()
    if digest != match.group(1).lower():
        raise PreflightError(f"ZIP/checksum mismatch: {symbol} {month}")
    year, number = map(int, month.split("-"))
    expected = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp() * 1000)
    required = monthrange(year, number)[1] * 288
    count = 0
    try:
        with ZipFile(io.BytesIO(body)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                    or members[0].file_size > 10_000_000 or members[0].flag_bits & 1):
                raise PreflightError(f"unexpected ZIP member: {symbol} {month}")
            with archive.open(members[0]) as raw:
                rows = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
                for fields in rows:
                    if count == 0 and fields and fields[0] == "open_time":
                        continue
                    if (len(fields) != 12 or fields[0] != str(expected)
                            or fields[6] != str(expected + 300_000 - 1)):
                        raise PreflightError(f"five-minute timestamp gap at row {count}: {symbol} {month}")
                    count += 1
                    expected += 300_000
    except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
        raise PreflightError(f"bad ZIP or timestamp CSV: {symbol} {month}") from exc
    if count != required:
        raise PreflightError(f"truncated five-minute grid ({count}/{required}): {symbol} {month}")
    return {"month": month, "market": "um", "symbol": symbol,
            "dataset": "5m", "sha256": digest, "bars": count}


def audit_all() -> tuple[list[dict[str, str | int]], dict[str, str]]:
    """Audit each symbol in sequence; stop at its first invalid month."""
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
                print(f"data-only: {symbol} full grid passed", flush=True)
            else:
                failures[symbol] = error
                print(f"data-only: {symbol} rejected at {len(rows)}/{len(months())} months: {error}", flush=True)
    complete.sort(key=lambda row: (row["month"], SYMBOLS.index(str(row["symbol"]))))
    return complete, failures


def main() -> None:
    rows, failures = audit_all()
    if failures:
        print(f"data-only preflight failed: {len(failures)} symbols; no universe or strategy is registered", flush=True)
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"data-only manifest: {OUTPUT}, {len(rows)} ZIP pins", flush=True)


if __name__ == "__main__":
    main()
