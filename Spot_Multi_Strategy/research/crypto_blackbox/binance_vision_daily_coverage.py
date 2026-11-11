"""Price-blind, read-only coverage audit for remote prospective daily archives."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import re

from .binance_vision_daily_prospective import (
    ACTIVATION_DATE, MAX_ZIP_BYTES, OUTPUT_ROOT, SYMBOLS, _archive_urls,
)


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MAX_SUMMARY_BYTES = 2 * 1024 * 1024


class CoverageError(ValueError):
    """A source file cannot support its stated daily-coverage claim."""


def _read_summary(path: Path, day: date) -> tuple[dict[str, tuple[int, int]], int]:
    try:
        with gzip.open(path, "rb") as file:
            raw = file.read(MAX_SUMMARY_BYTES + 1)
        if len(raw) > MAX_SUMMARY_BYTES:
            raise CoverageError("summary exceeds frozen size limit")
        body = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoverageError(f"unreadable prospective summary: {day}") from exc
    if (not isinstance(body, dict) or body.get("schema") != 1
            or body.get("source") != "binance_vision_spot_daily_aggTrades"
            or body.get("day") != day.isoformat()
            or body.get("independent_executable_oos") is not False):
        raise CoverageError(f"prospective summary contract changed: {day}")
    try:
        captured = datetime.fromisoformat(body["captured_at_utc"])
    except (TypeError, ValueError, KeyError) as exc:
        raise CoverageError(f"missing actual capture timestamp: {day}") from exc
    if (captured.tzinfo is None or captured.utcoffset() != timedelta(0)
            or captured.date() <= day):
        raise CoverageError(f"source captured before UTC day completed: {day}")
    sources = body.get("sources")
    if (not isinstance(sources, list) or len(sources) != len(SYMBOLS)
            or not all(isinstance(x, dict) for x in sources)
            or [x.get("symbol") for x in sources] != list(SYMBOLS)):
        raise CoverageError(f"frozen symbol universe incomplete: {day}")
    ids: dict[str, tuple[int, int]] = {}
    for source in sources:
        symbol = source["symbol"]
        url, _, _ = _archive_urls(day, symbol)
        digest, bytes_, rows, hours = (source.get("zip_sha256"), source.get("zip_bytes"),
                                       source.get("rows"), source.get("hours"))
        if (source.get("url") != url or not isinstance(digest, str)
                or not HEX64.fullmatch(digest) or type(bytes_) is not int
                or not 0 < bytes_ <= MAX_ZIP_BYTES or type(rows) is not int
                or rows < 24 or not isinstance(hours, list) or len(hours) != 24):
            raise CoverageError(f"source archive metadata invalid: {day} {symbol}")
        summed = 0
        for hour, item in enumerate(hours):
            if not isinstance(item, dict):
                raise CoverageError(f"hourly source row invalid: {day} {symbol}")
            count, buys, sells = (item.get("count"), item.get("buy_count"),
                                  item.get("sell_count"))
            first_id, last_id = item.get("first_id"), item.get("last_id")
            notional = (item.get("buy_quote_micros"), item.get("sell_quote_micros"))
            if (item.get("hour") != hour or item.get("status") != "observed"
                    or any(type(x) is not int or x < 0 for x in
                           (count, buys, sells, first_id, last_id, *notional))
                    or count == 0 or count != buys + sells
                    or last_id - first_id + 1 != count
                    or (hour and first_id != hours[hour - 1]["last_id"] + 1)):
                raise CoverageError(f"hourly source continuity invalid: {day} {symbol} {hour}")
            summed += count
        if rows != summed:
            raise CoverageError(f"archive row count mismatch: {day} {symbol}")
        ids[symbol] = (hours[0]["first_id"], hours[-1]["last_id"])
    return ids, int(captured.timestamp() * 1000)


def audit(root: Path = OUTPUT_ROOT, *, as_of: date | None = None) -> dict:
    current = datetime.now(timezone.utc)
    today = current.date() if as_of is None else as_of
    if type(today) is not date:
        raise ValueError("as_of must be a UTC date")
    expected = max(0, (today - ACTIVATION_DATE).days)
    cutoff_ms = int((current if as_of is None else datetime.combine(
        today + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)).timestamp() * 1000)
    complete = 0
    longest = run = 0
    longest_start = run_start = None
    first_bad: list[str] = []
    previous_ids = None
    prior_complete = False
    total_bytes = 0
    for offset in range(expected):
        day = ACTIVATION_DATE + timedelta(days=offset)
        path = root / f"{day.isoformat()}.json.gz"
        valid = False
        if path.is_file():
            total_bytes += path.stat().st_size
            ids, captured_ms = _read_summary(path, day)
            if captured_ms >= cutoff_ms:
                raise CoverageError(f"capture timestamp exceeds audit cutoff: {day}")
            valid = not (prior_complete and previous_ids is not None and
                         any(ids[symbol][0] != previous_ids[symbol][1] + 1
                             for symbol in SYMBOLS))
            previous_ids = ids
        if valid:
            complete += 1
            if run == 0:
                run_start = day
            run += 1
            if run > longest:
                longest, longest_start = run, run_start
        else:
            if len(first_bad) < 10:
                first_bad.append(day.isoformat())
            run = 0
            previous_ids = None
        prior_complete = valid
    longest_end = longest_start + timedelta(days=longest) if longest_start else None
    five_years = False
    if longest_start and longest_end:
        try:
            anniversary = longest_start.replace(year=longest_start.year + 5)
        except ValueError:
            anniversary = longest_start.replace(year=longest_start.year + 5,
                                                month=3, day=1)
        five_years = longest_end >= anniversary
    return {"activation_date": ACTIVATION_DATE.isoformat(), "as_of_date": today.isoformat(),
            "expected_days": expected, "complete_days": complete,
            "missing_or_bad_days": expected - complete, "first_bad_days": first_bad,
            "longest_complete_run_days": longest,
            "longest_complete_start": longest_start.isoformat() if longest_start else None,
            "longest_complete_end_exclusive": longest_end.isoformat() if longest_end else None,
            "five_year_continuous_source": five_years,
            "independent_executable_oos_proven": False,
            "compressed_summary_bytes": total_bytes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
