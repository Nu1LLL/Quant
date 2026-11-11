"""Append-only, next-day Binance Vision spot aggTrade source summaries.

This collects public source data, never signals, backtests, account information
or orders. A date is captured only after the frozen activation date and the
complete UTC day. Each output contains its actual capture time and archive
hashes; it cannot claim intraday availability before the archive was fetched.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile


ACTIVATION_DATE = date(2026, 11, 13)
SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "SOLUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT", "DOTUSDT", "AVAXUSDT",
)
OUTPUT_ROOT = Path("Spot_Multi_Strategy/research/prospective/binance_vision_daily")
BASE = "https://data.binance.vision/data/spot/daily/aggTrades"
MAX_ZIP_BYTES = 128 * 1024 * 1024
MAX_CSV_BYTES = 512 * 1024 * 1024
MAX_ROWS = 10_000_000
HEX64 = re.compile(r"[0-9a-fA-F]{64}\Z")


class SourceFailure(ValueError):
    """Archive, checksum, schema, timing or append-only failure."""


class ArchivePending(SourceFailure):
    """The exchange has not published a required daily archive yet."""


def _fetch(url: str, maximum: int) -> bytes:
    try:
        with urlopen(Request(url, headers={"User-Agent": "MiniMedallion-Source-Audit/1"}),
                     timeout=45) as response:
            raw = response.read(maximum + 1)
    except HTTPError as exc:
        if exc.code == 404:
            raise ArchivePending("official daily archive not yet published") from exc
        raise SourceFailure(f"official archive HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise SourceFailure(f"official archive transport: {type(exc).__name__}") from exc
    if len(raw) > maximum:
        raise SourceFailure("official archive exceeds frozen size limit")
    return raw


def _archive_urls(day: date, symbol: str) -> tuple[str, str, str]:
    filename = f"{symbol}-aggTrades-{day.isoformat()}.zip"
    url = f"{BASE}/{symbol}/{filename}"
    return url, url + ".CHECKSUM", filename


def _check_checksum(raw: bytes, checksum: bytes, filename: str) -> str:
    try:
        fields = checksum.decode("ascii").strip().split()
    except UnicodeDecodeError as exc:
        raise SourceFailure("non-ASCII official checksum") from exc
    if len(fields) != 2 or not HEX64.fullmatch(fields[0]) or fields[1].lstrip("*") != filename:
        raise SourceFailure("official checksum schema or filename mismatch")
    digest = sha256(raw).hexdigest()
    if digest.lower() != fields[0].lower():
        raise SourceFailure("official archive SHA-256 mismatch")
    return digest


def _to_positive_decimal(value: str) -> Decimal:
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise SourceFailure("nondecimal trade price/quantity") from exc
    if not number.is_finite() or number <= 0:
        raise SourceFailure("nonpositive trade price/quantity")
    return number


def _summarize_csv(csv_file, day: date) -> tuple[list[dict], int]:
    start_us = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) * 1_000_000
    hourly = [{"hour": hour, "status": "empty", "count": 0, "buy_count": 0,
               "sell_count": 0, "buy_quote_micros": 0, "sell_quote_micros": 0,
               "first_id": None, "last_id": None} for hour in range(24)]
    previous_id = None
    row_count = 0
    try:
        reader = csv.reader(io.TextIOWrapper(csv_file, encoding="utf-8", newline=""))
        for row in reader:
            row_count += 1
            if row_count > MAX_ROWS or len(row) != 8:
                raise SourceFailure("aggTrades row count or 8-field schema invalid")
            try:
                agg_id, first_trade_id, last_trade_id, stamp = (
                    int(row[i]) for i in (0, 3, 4, 5))
            except ValueError as exc:
                raise SourceFailure("invalid aggregate/trade ID or timestamp") from exc
            if (agg_id < 0 or first_trade_id < 0 or last_trade_id < first_trade_id
                    or (previous_id is not None and agg_id != previous_id + 1)
                    or not start_us <= stamp < start_us + 86_400_000_000):
                raise SourceFailure("aggregate ID continuity or UTC-day timestamp invalid")
            if row[6] not in ("True", "False") or row[7] not in ("True", "False"):
                raise SourceFailure("aggregate trade Boolean fields invalid")
            price = _to_positive_decimal(row[1])
            quantity = _to_positive_decimal(row[2])
            micros = int((price * quantity * 1_000_000).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP))
            if micros <= 0 or micros >= 2**63:
                raise SourceFailure("trade quote notional outside storage range")
            hour = (stamp - start_us) // 3_600_000_000
            bucket = hourly[hour]
            bucket["status"] = "observed"
            bucket["count"] += 1
            if row[6] == "True":  # buyer is maker, hence sell aggressor
                bucket["sell_count"] += 1
                bucket["sell_quote_micros"] += micros
            else:
                bucket["buy_count"] += 1
                bucket["buy_quote_micros"] += micros
            if bucket["first_id"] is None:
                bucket["first_id"] = agg_id
            bucket["last_id"] = agg_id
            previous_id = agg_id
    except (UnicodeError, OverflowError, zipfile.BadZipFile) as exc:
        raise SourceFailure(f"invalid aggTrades CSV: {type(exc).__name__}") from exc
    if row_count == 0 or any(bucket["status"] != "observed" for bucket in hourly):
        raise SourceFailure("empty archive or incomplete 24-hour symbol coverage")
    return hourly, row_count


def _summarize_archive(day: date, symbol: str, fetch=_fetch) -> dict:
    url, checksum_url, filename = _archive_urls(day, symbol)
    raw = fetch(url, MAX_ZIP_BYTES)
    checksum = fetch(checksum_url, 4096)
    digest = _check_checksum(raw, checksum, filename)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != filename[:-4] + ".csv"
                    or members[0].file_size > MAX_CSV_BYTES):
                raise SourceFailure("unexpected ZIP member or decompressed size")
            with archive.open(members[0]) as csv_file:
                hours, count = _summarize_csv(csv_file, day)
    except zipfile.BadZipFile as exc:
        raise SourceFailure("invalid official ZIP") from exc
    return {"symbol": symbol, "url": url, "zip_sha256": digest,
            "zip_bytes": len(raw), "rows": count, "hours": hours}


def capture_day(day: date, output_root: Path, *, fetch=_fetch,
                captured_at: datetime | None = None, code_commit: str | None = None) -> Path:
    if type(day) is not date or day < ACTIVATION_DATE:
        raise ValueError("source date precedes frozen prospective activation")
    now = datetime.now(timezone.utc) if captured_at is None else captured_at
    if now.tzinfo is None or now.utcoffset() != timedelta(0) or day >= now.date():
        raise ValueError("only completed past UTC days can be captured")
    target = output_root / f"{day.isoformat()}.json.gz"
    if target.exists():
        return target
    sources = [_summarize_archive(day, symbol, fetch) for symbol in SYMBOLS]
    sealed_at = datetime.now(timezone.utc) if captured_at is None else captured_at
    body = {"schema": 1, "source": "binance_vision_spot_daily_aggTrades",
            "day": day.isoformat(), "captured_at_utc": sealed_at.isoformat(),
            "code_commit": code_commit, "independent_executable_oos": False,
            "sources": sources}
    compressed = gzip.compress(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode(), mtime=0)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_root, prefix=".capture-", delete=False) as file:
        temporary = Path(file.name)
        file.write(compressed)
        file.flush()
        os.fsync(file.fileno())
    try:
        os.link(temporary, target)  # exclusive: a prior capture is never replaced
    except FileExistsError as exc:
        raise SourceFailure("prospective source day already captured concurrently") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--lookback-days", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.lookback_days <= 14:
        parser.error("lookback-days must be 1..14")
    today = datetime.now(timezone.utc).date()
    first = max(ACTIVATION_DATE, today - timedelta(days=args.lookback_days))
    for offset in range((today - first).days):
        day = first + timedelta(days=offset)
        target = args.output_root / f"{day.isoformat()}.json.gz"
        if target.exists():
            continue
        try:
            capture_day(day, args.output_root, code_commit=os.environ.get("GITHUB_SHA"))
            print(f"captured {day.isoformat()}")
        except ArchivePending:
            print(f"pending official archive {day.isoformat()}")


if __name__ == "__main__":
    main()
