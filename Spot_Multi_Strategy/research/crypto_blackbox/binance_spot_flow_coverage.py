"""Read-only, price-blind coverage audit of prospective Binance spot flow."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time

from .binance_spot_flow_prospective import (
    ACTIVATION_MS, GRACE_MS, HOUR_MS, JOURNAL, MAX_DB_BYTES, SYMBOLS,
)


FIVE_YEAR_END_MS = int(datetime(2031, 11, 9, tzinfo=timezone.utc).timestamp() * 1000)
STATUSES = frozenset(("observed", "empty", "invalid", "unavailable"))


class CoverageError(ValueError):
    """The journal cannot support the claimed source-coverage status."""


def audit(path: Path = JOURNAL, *, as_of_ms: int | None = None) -> dict:
    now = time.time_ns() // 1_000_000 if as_of_ms is None else as_of_ms
    if type(now) is not int or now <= 0:
        raise ValueError("as_of_ms must be a positive UTC epoch millisecond")
    end = max(ACTIVATION_MS, ((now - GRACE_MS) // HOUR_MS) * HOUR_MS)
    expected = (end - ACTIVATION_MS) // HOUR_MS
    location = Path(path)
    base = {
        "as_of_ms": now, "activation_ms": ACTIVATION_MS,
        "last_due_window_end_ms": end,
        "five_year_end_ms": FIVE_YEAR_END_MS,
        "expected_hours_per_symbol": expected,
        "symbol_count": len(SYMBOLS),
        "five_year_independent_oos_proven": False,
    }
    if not location.is_file():
        return {**base, "status": ("awaiting_first_hour" if expected == 0
                                    else "journal_missing"),
                "observed_hours_all_symbols": 0,
                "longest_complete_run_hours": 0,
                "longest_complete_start_ms": None,
                "longest_complete_end_ms": None,
                "five_year_continuous_source": False,
                "missing_rows": expected * len(SYMBOLS),
                "first_bad_hours_ms": [ACTIVATION_MS + i * HOUR_MS
                                       for i in range(min(expected, 10))]}
    size = sum(p.stat().st_size for p in (location, Path(str(location) + "-wal"))
               if p.exists())
    if size > MAX_DB_BYTES:
        raise CoverageError("source journal exceeds its 96 MiB cap")
    try:
        with closing(sqlite3.connect(location.resolve().as_uri() + "?mode=ro",
                                     uri=True, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            contract_row = db.execute(
                "SELECT contract FROM source_contract WHERE id=1").fetchone()
            if contract_row is None:
                raise CoverageError("source contract missing")
            contract = json.loads(contract_row[0])
            if (contract.get("activation_ms") != ACTIVATION_MS
                    or tuple(contract.get("symbols", ())) != SYMBOLS
                    or contract.get("schema") != 1
                    or contract.get("source") != "binance_spot_aggTrade"):
                raise CoverageError("source contract changed")
            incidents = set()
            incident_rows = db.execute("""
                SELECT symbol,window_start_ms,detected_at_ms,reason
                FROM source_incident WHERE window_start_ms>=? AND window_start_ms<?
                  AND detected_at_ms<=?
            """, (ACTIVATION_MS, end, now))
            for incident in incident_rows:
                symbol, start, detected, reason = incident
                if (symbol not in SYMBOLS or type(start) is not int
                        or (start - ACTIVATION_MS) % HOUR_MS
                        or type(detected) is not int
                        or detected < start + HOUR_MS + GRACE_MS
                        or reason != "late_after_seal"):
                    raise CoverageError("invalid append-only source incident")
                incidents.add((symbol, start))
            rows = db.execute("""
                SELECT symbol,window_start_ms,recorded_at_ms,status,count,
                       buys,sells,buy_micros,sell_micros,first_id,last_id,
                       first_received_ms,last_received_ms
                FROM hourly_flow WHERE window_start_ms>=? AND window_start_ms<?
                  AND recorded_at_ms<=?
                ORDER BY window_start_ms,symbol
            """, (ACTIVATION_MS, end, now))
            seen: dict[int, set[str]] = {}
            observed: dict[int, int] = {}
            status_counts = {symbol: Counter() for symbol in SYMBOLS}
            previous_id: dict[str, int] = {}
            total_rows = 0
            for row in rows:
                symbol, start, recorded, status = (row["symbol"], row["window_start_ms"],
                                                   row["recorded_at_ms"], row["status"])
                count, buys, sells = row["count"], row["buys"], row["sells"]
                ids = row["first_id"], row["last_id"]
                stamps = row["first_received_ms"], row["last_received_ms"]
                if (symbol not in SYMBOLS or type(start) is not int
                        or start < ACTIVATION_MS or start >= end
                        or (start - ACTIVATION_MS) % HOUR_MS
                        or type(recorded) is not int
                        or not start + HOUR_MS + GRACE_MS <= recorded <= now
                        or status not in STATUSES
                        or any(type(row[k]) is not int or row[k] < 0 for k in
                               ("count", "buys", "sells", "buy_micros", "sell_micros"))
                        or count != buys + sells
                        or (count == 0 and any(v is not None for v in (*ids, *stamps)))
                        or (count > 0 and (any(type(v) is not int or v < 0 for v in (*ids, *stamps))
                                           or ids[0] > ids[1] or stamps[0] > stamps[1]))
                        or (status == "observed" and count == 0)
                        or (status == "observed" and ids[1] - ids[0] + 1 != count)
                        or (status == "empty" and count != 0)):
                    raise CoverageError("invalid prospective source row")
                if status == "observed" and (symbol, start) not in incidents:
                    prior_id = previous_id.get(symbol)
                    if prior_id is not None and ids[0] != prior_id + 1:
                        raise CoverageError("aggregate-trade ID gap across observed hours")
                    previous_id[symbol] = ids[1]
                elif status in {"invalid", "unavailable"} or (symbol, start) in incidents:
                    previous_id.pop(symbol, None)
                slot = seen.setdefault(start, set())
                if symbol in slot:
                    raise CoverageError("duplicate source symbol/hour")
                slot.add(symbol)
                observed[start] = observed.get(start, 0) + (
                    status == "observed" and (symbol, start) not in incidents)
                status_counts[symbol][status] += 1
                if (symbol, start) in incidents:
                    status_counts[symbol]["postseal_invalidated"] += 1
                total_rows += 1
            if any(symbol not in seen.get(start, ()) for symbol, start in incidents):
                raise CoverageError("source incident has no sealed hourly row")
            db.execute("ROLLBACK")
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        raise CoverageError("journal unreadable or schema changed") from exc
    missing = expected * len(SYMBOLS) - total_rows
    if missing < 0:
        raise CoverageError("more rows than expected UTC grid")
    for symbol in SYMBOLS:
        status_counts[symbol]["missing"] = expected - sum(status_counts[symbol].values())
    all_observed = sum(observed.get(ACTIVATION_MS + i * HOUR_MS, 0) == len(SYMBOLS)
                       for i in range(expected))
    longest = current = 0
    longest_start = current_start = None
    for i in range(expected):
        start = ACTIVATION_MS + i * HOUR_MS
        if observed.get(start, 0) == len(SYMBOLS):
            if current == 0:
                current_start = start
            current += 1
            if current > longest:
                longest, longest_start = current, current_start
        else:
            current = 0
    longest_end = None if longest_start is None else longest_start + longest * HOUR_MS
    five_year_continuous = False
    if longest_start is not None:
        beginning = datetime.fromtimestamp(longest_start / 1000, tz=timezone.utc)
        try:
            anniversary = beginning.replace(year=beginning.year + 5)
        except ValueError:
            anniversary = beginning.replace(year=beginning.year + 5, month=3, day=1)
        five_year_continuous = longest_end >= int(anniversary.timestamp() * 1000)
    first_five_hours = (FIVE_YEAR_END_MS - ACTIVATION_MS) // HOUR_MS
    first_five_observed = sum(
        observed.get(ACTIVATION_MS + i * HOUR_MS, 0) == len(SYMBOLS)
        for i in range(min(expected, first_five_hours)))
    first_bad = [ACTIVATION_MS + i * HOUR_MS for i in range(expected)
                 if observed.get(ACTIVATION_MS + i * HOUR_MS, 0) != len(SYMBOLS)][:10]
    return {**base, "status": "audited", "database_bytes": size,
            "observed_hours_all_symbols": all_observed,
            "longest_complete_run_hours": longest,
            "longest_complete_start_ms": longest_start,
            "longest_complete_end_ms": longest_end,
            "five_year_continuous_source": five_year_continuous,
            "postseal_incident_hours": len(incidents),
            "missing_rows": missing,
            "first_bad_hours_ms": first_bad,
            "status_counts": {symbol: dict(sorted(count.items()))
                              for symbol, count in status_counts.items()},
            "first_five_year_source_coverage_complete": (
                end >= FIVE_YEAR_END_MS and first_five_observed == first_five_hours),
            "note": "Observed is local continuity only; no exchange losslessness or executable fill proof."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=JOURNAL)
    parser.add_argument("--as-of-ms", type=int)
    args = parser.parse_args()
    print(json.dumps(audit(args.journal, as_of_ms=args.as_of_ms), sort_keys=True))


if __name__ == "__main__":
    main()
