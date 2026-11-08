"""Read-only continuity audit of the prospective Hyperliquid source journal.

This is source availability evidence only. It never reads prices, outcomes,
wallet holdings, or orders and cannot approve a trading hypothesis.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import time

from .hyperliquid_flow_prospective_journal import (
    ACTIVATION_MS, FIVE_MIN_MS, HOUR_MS, JOURNAL,
    MAX_FRESH_DELAY_MS, MAX_JOURNAL_BYTES, SYMBOLS,
)


FIVE_YEAR_END_MS = int(datetime(2031, 11, 9, tzinfo=timezone.utc).timestamp() * 1000)
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_STATUSES = frozenset(("timely", "late", "unavailable", "empty", "invalid"))


class FlowCoverageError(ValueError):
    """A journal exists, but its contents cannot support an as-of audit."""


def _window_end(as_of_ms: int) -> int:
    if type(as_of_ms) is not int or as_of_ms <= 0:
        raise ValueError("as_of_ms must be a positive integer UTC epoch millisecond")
    return max(ACTIVATION_MS, ((as_of_ms - FIVE_MIN_MS) // HOUR_MS) * HOUR_MS)


def _empty_report(as_of_ms: int, end_ms: int, status: str) -> dict:
    expected = (end_ms - ACTIVATION_MS) // HOUR_MS
    first_five = min(end_ms, FIVE_YEAR_END_MS)
    return {
        "status": status,
        "as_of_ms": as_of_ms,
        "activation_ms": ACTIVATION_MS,
        "last_complete_window_end_ms": end_ms,
        "five_year_end_ms": FIVE_YEAR_END_MS,
        "expected_hours": expected,
        "observed_hours_both_symbols": 0,
        "timely_hours_both_symbols": 0,
        "bad_hours": expected,
        "missing_rows": expected * len(SYMBOLS),
        "status_counts": {symbol: {"missing": expected} for symbol in SYMBOLS},
        "first_bad_windows_ms": [ACTIVATION_MS + i * HOUR_MS
                                 for i in range(min(expected, 10))],
        "max_consecutive_bad_hours": expected,
        "first_five_year_hours_expected": (first_five - ACTIVATION_MS) // HOUR_MS,
        "first_five_year_hours_timely": 0,
        "first_five_year_source_coverage_complete": False,
        "five_year_independent_oos_proven": False,
        "data_digest_sha256": sha256(b"").hexdigest(),
    }


def audit_coverage(path: Path = JOURNAL, *, as_of_ms: int | None = None) -> dict:
    """Audit every due hour for both fixed symbols without creating a journal.

    `timely` is only a source-observation classification. Even five years of
    timely rows cannot prove lossless WebSocket delivery or executable fills.
    """
    observed = time.time_ns() // 1_000_000 if as_of_ms is None else as_of_ms
    end = _window_end(observed)
    if end == ACTIVATION_MS and observed < ACTIVATION_MS + HOUR_MS + FIVE_MIN_MS:
        return _empty_report(observed, end, "awaiting_first_complete_hour")
    location = Path(path)
    if not location.is_file():
        return _empty_report(observed, end, "journal_missing")
    total_bytes = sum(p.stat().st_size for p in (location, Path(str(location) + "-wal"))
                      if p.exists())
    if total_bytes > MAX_JOURNAL_BYTES:
        raise FlowCoverageError("prospective journal exceeds its registered 64 MiB cap")
    try:
        with closing(sqlite3.connect(location.resolve().as_uri() + "?mode=ro",
                                     uri=True, timeout=5)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            rows = connection.execute("""
                SELECT symbol,window_start_ms,window_end_ms,observed_at_ms,
                       status,source_health_at_ms,trade_count,buy_count,
                       sell_count,buy_cents,sell_cents,source_digest
                FROM hourly_flow
                WHERE window_start_ms>=? AND window_start_ms<?
                  AND observed_at_ms<=?
                ORDER BY window_start_ms,symbol
            """, (ACTIVATION_MS, end, observed))
            by_hour: dict[int, dict[str, str]] = {}
            counters = {symbol: Counter() for symbol in SYMBOLS}
            digest = sha256()
            row_count = 0
            for row in rows:
                start = row["window_start_ms"]
                symbol = row["symbol"]
                finish = row["window_end_ms"]
                seen_at = row["observed_at_ms"]
                health_at = row["source_health_at_ms"]
                status = row["status"]
                count, buys, sells = (row["trade_count"], row["buy_count"],
                                       row["sell_count"])
                buy_cents, sell_cents = row["buy_cents"], row["sell_cents"]
                source_hash = row["source_digest"]
                if (symbol not in SYMBOLS or type(start) is not int
                        or start < ACTIVATION_MS or start >= end
                        or (start - ACTIVATION_MS) % HOUR_MS
                        or finish != start + HOUR_MS
                        or type(seen_at) is not int
                        or not finish + FIVE_MIN_MS <= seen_at <= observed
                        or status not in _STATUSES
                        or any(type(n) is not int or n < 0 for n in
                               (count, buys, sells, buy_cents, sell_cents))
                        or count != buys + sells
                        or not isinstance(source_hash, str)
                        or not _HEX.fullmatch(source_hash)):
                    raise FlowCoverageError("invalid prospective source row")
                if status in ("timely", "late"):
                    if (count == 0 or health_at is None
                            or type(health_at) is not int
                            or not 0 <= seen_at - health_at <= 600_000):
                        raise FlowCoverageError("claimed healthy source row has no fresh evidence")
                    if status == "timely" and seen_at - finish > MAX_FRESH_DELAY_MS:
                        raise FlowCoverageError("claimed timely row was captured late")
                    if status == "late" and seen_at - finish <= MAX_FRESH_DELAY_MS:
                        raise FlowCoverageError("claimed late row was captured on time")
                elif count or buys or sells or buy_cents or sell_cents:
                    raise FlowCoverageError("unavailable/empty/invalid row has trade data")
                elif status == "empty" and health_at is None:
                    raise FlowCoverageError("empty source row lacks health timestamp")
                slot = by_hour.setdefault(start, {})
                if symbol in slot:
                    raise FlowCoverageError("duplicate source symbol/hour")
                slot[symbol] = status
                counters[symbol][status] += 1
                digest.update((f"{symbol},{start},{finish},{seen_at},{status},"
                               f"{health_at},{count},{buys},{sells},{buy_cents},"
                               f"{sell_cents},{source_hash}\n").encode("ascii"))
                row_count += 1
            connection.execute("ROLLBACK")
    except sqlite3.Error as exc:
        raise FlowCoverageError("prospective journal unreadable or schema changed") from exc

    expected = (end - ACTIVATION_MS) // HOUR_MS
    first_five_expected = (min(end, FIVE_YEAR_END_MS) - ACTIVATION_MS) // HOUR_MS
    both = timely = first_five_timely = bad = max_bad = streak = missing = 0
    first_bad: list[int] = []
    for offset in range(expected):
        start = ACTIVATION_MS + offset * HOUR_MS
        slot = by_hour.get(start, {})
        all_present = all(symbol in slot for symbol in SYMBOLS)
        all_timely = all(slot.get(symbol) == "timely" for symbol in SYMBOLS)
        both += all_present
        timely += all_timely
        if start < FIVE_YEAR_END_MS:
            first_five_timely += all_timely
        if not all_timely:
            bad += 1
            streak += 1
            max_bad = max(max_bad, streak)
            if len(first_bad) < 10:
                first_bad.append(start)
        else:
            streak = 0
        for symbol in SYMBOLS:
            if symbol not in slot:
                counters[symbol]["missing"] += 1
                missing += 1
    if row_count + missing != expected * len(SYMBOLS):
        raise FlowCoverageError("journal row count disagrees with hourly grid")
    first_five_complete = (end >= FIVE_YEAR_END_MS
                           and first_five_timely == (FIVE_YEAR_END_MS - ACTIVATION_MS) // HOUR_MS)
    return {
        "status": "audited",
        "as_of_ms": observed,
        "activation_ms": ACTIVATION_MS,
        "last_complete_window_end_ms": end,
        "five_year_end_ms": FIVE_YEAR_END_MS,
        "expected_hours": expected,
        "observed_hours_both_symbols": both,
        "timely_hours_both_symbols": timely,
        "bad_hours": bad,
        "missing_rows": missing,
        "status_counts": {symbol: dict(sorted(counter.items()))
                          for symbol, counter in counters.items()},
        "first_bad_windows_ms": first_bad,
        "max_consecutive_bad_hours": max_bad,
        "first_five_year_hours_expected": first_five_expected,
        "first_five_year_hours_timely": first_five_timely,
        "first_five_year_source_coverage_complete": first_five_complete,
        "five_year_independent_oos_proven": False,
        "data_digest_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=JOURNAL)
    parser.add_argument("--as-of-ms", type=int)
    args = parser.parse_args()
    print(json.dumps(audit_coverage(args.journal, as_of_ms=args.as_of_ms),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
