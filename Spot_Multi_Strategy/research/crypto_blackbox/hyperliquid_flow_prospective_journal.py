"""Small, append-only, point-in-time Hyperliquid trade-flow source journal.

This records source observations, not a trading hypothesis, PnL, wallet
position, or order. It deliberately never imports pre-activation history.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sqlite3
import time


HOUR_MS = 3_600_000
FIVE_MIN_MS = 300_000
ACTIVATION_MS = 1_794_182_400_000  # 2026-11-09 00:00:00 UTC
MAX_FRESH_DELAY_MS = 900_000
MAX_SOURCE_LAG_MS = 300_000
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
SYMBOLS = ("BTC", "ETH")
SOURCE = Path.home() / "Library/Application Support/WhaleWatcher/data/onchain_state/whale_robot.sqlite"
JOURNAL = Path.home() / "Library/Application Support/MiniMedallion/hyperliquid_flow_prospective.sqlite"


@dataclass(frozen=True)
class Capture:
    symbol: str
    window_start_ms: int
    observed_at_ms: int
    status: str
    trade_count: int
    buy_cents: int
    sell_cents: int


def _source_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _journal_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS hourly_flow (
            symbol TEXT NOT NULL CHECK(symbol IN ('BTC','ETH')),
            window_start_ms INTEGER NOT NULL,
            window_end_ms INTEGER NOT NULL,
            observed_at_ms INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN
                ('fresh','late','unavailable','empty','invalid')),
            source_health_at_ms INTEGER,
            trade_count INTEGER NOT NULL CHECK(trade_count>=0),
            buy_count INTEGER NOT NULL CHECK(buy_count>=0),
            sell_count INTEGER NOT NULL CHECK(sell_count>=0),
            buy_cents INTEGER NOT NULL CHECK(buy_cents>=0),
            sell_cents INTEGER NOT NULL CHECK(sell_cents>=0),
            source_digest TEXT NOT NULL CHECK(length(source_digest)=64),
            PRIMARY KEY(symbol,window_start_ms),
            CHECK(window_end_ms=window_start_ms+3600000),
            CHECK(window_start_ms>=1794182400000),
            CHECK(observed_at_ms>=window_end_ms+300000),
            CHECK(trade_count=buy_count+sell_count)
        )
    """)
    connection.execute("""
        CREATE TRIGGER IF NOT EXISTS hourly_flow_no_update
        BEFORE UPDATE ON hourly_flow BEGIN SELECT RAISE(ABORT,'append-only'); END
    """)
    connection.execute("""
        CREATE TRIGGER IF NOT EXISTS hourly_flow_no_delete
        BEFORE DELETE ON hourly_flow BEGIN SELECT RAISE(ABORT,'append-only'); END
    """)
    return connection


def _journal_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in (path, Path(str(path) + "-wal")) if p.exists())


def _window(now_ms: int) -> tuple[int, int] | None:
    end = ((now_ms - FIVE_MIN_MS) // HOUR_MS) * HOUR_MS
    start = end - HOUR_MS
    if start < ACTIVATION_MS:
        return None
    return start, end


def _cents(value: object) -> int:
    number = Decimal(str(value))
    if not number.is_finite() or number <= 0:
        raise ValueError("invalid trade notional")
    return int((number * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _read_symbol(source: sqlite3.Connection, symbol: str, start: int,
                 end: int, observed: int) -> tuple[int, int, int, int, int, str]:
    digest = hashlib.sha256()
    count = buys = sells = buy_cents = sell_cents = 0
    rows = source.execute("""
        SELECT row_id,trade_id,timestamp_ms,side,notional_usd,inserted_at_ms,
               venue,market_type,confirmation_status
        FROM market_flow_trades
        WHERE symbol=? AND timestamp_ms>=? AND timestamp_ms<?
        ORDER BY timestamp_ms,row_id
    """, (symbol, start, end))
    for row in rows:
        timestamp = int(row["timestamp_ms"])
        inserted = int(row["inserted_at_ms"])
        if (row["venue"] != "hyperliquid" or row["market_type"] != "perpetual"
                or row["confirmation_status"] != "executed"
                or row["side"] not in ("long", "short")
                or inserted < timestamp or inserted - timestamp > MAX_SOURCE_LAG_MS
                or inserted > observed):
            raise ValueError("source row violates real-time trade contract")
        cents = _cents(row["notional_usd"])
        canonical = (int(row["row_id"]), str(row["trade_id"]), timestamp,
                     row["side"], cents, inserted)
        digest.update(json.dumps(canonical, separators=(",", ":")).encode() + b"\n")
        count += 1
        if row["side"] == "long":  # Hyperliquid B: aggressing buy, not a long position.
            buys += 1
            buy_cents += cents
        else:
            sells += 1
            sell_cents += cents
    return count, buys, sells, buy_cents, sell_cents, digest.hexdigest()


def capture_once(source_path: Path = SOURCE, journal_path: Path = JOURNAL,
                 *, now_ms: int | None = None) -> tuple[Capture, ...]:
    """Capture only the most recent hour after its five-minute safety delay.

    Missed hours remain visible as gaps. Never backfill them as live evidence.
    Repeated launches are idempotent, and results cannot be rewritten.
    """
    observed = time.time_ns() // 1_000_000 if now_ms is None else int(now_ms)
    window = _window(observed)
    if window is None:
        return ()
    start, end = window
    with closing(_journal_connection(Path(journal_path))) as journal:
        previous = journal.execute(
            "SELECT COUNT(*) FROM hourly_flow WHERE window_start_ms=?", (start,)
        ).fetchone()[0]
        if previous == len(SYMBOLS):
            return ()
        if _journal_bytes(Path(journal_path)) >= MAX_JOURNAL_BYTES:
            raise RuntimeError("prospective flow journal reached its 64 MiB limit")
        health_time: int | None = None
        source_ok = False
        source_invalid = False
        measurements: dict[str, tuple[int, int, int, int, int, str]] = {}
        try:
            with closing(_source_connection(Path(source_path))) as source:
                source.execute("BEGIN")  # One immutable source snapshot for both coins.
                health = source.execute("""
                    SELECT status,healthy,connected,last_event_at_ms,updated_at_ms
                    FROM market_flow_source_health WHERE source='hyperliquid_perpetual'
                """).fetchone()
                if health is not None:
                    health_time = int(health["updated_at_ms"])
                    source_ok = (health["status"] == "streaming"
                                 and int(health["healthy"]) == 1
                                 and int(health["connected"]) == 1
                                 and health["last_event_at_ms"] is not None
                                 and int(health["last_event_at_ms"]) >= end - FIVE_MIN_MS
                                 and 0 <= observed - health_time <= 600_000)
                if source_ok:
                    for symbol in SYMBOLS:
                        measurements[symbol] = _read_symbol(
                            source, symbol, start, end, observed)
                source.execute("ROLLBACK")
        except (ValueError, ArithmeticError, TypeError):
            source_invalid = True
            source_ok = False
            measurements = {}
        except (OSError, sqlite3.Error):
            source_ok = False
            measurements = {}
        appended: list[Capture] = []
        with journal:
            for symbol in SYMBOLS:
                if journal.execute("""
                    SELECT 1 FROM hourly_flow WHERE symbol=? AND window_start_ms=?
                """, (symbol, start)).fetchone():
                    continue
                count, buys, sells, buy_cents, sell_cents, digest = measurements.get(
                    symbol, (0, 0, 0, 0, 0, hashlib.sha256(b"").hexdigest()))
                status = ("invalid" if source_invalid else
                          "unavailable" if not source_ok else
                          "empty" if count == 0 else
                          "late" if observed - end > MAX_FRESH_DELAY_MS else
                          "fresh")
                journal.execute("""
                    INSERT INTO hourly_flow VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (symbol, start, end, observed, status, health_time,
                      count, buys, sells, buy_cents, sell_cents, digest))
                appended.append(Capture(symbol, start, observed, status,
                                        count, buy_cents, sell_cents))
        return tuple(appended)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--journal", type=Path, default=JOURNAL)
    args = parser.parse_args()
    print(json.dumps([item.__dict__ for item in capture_once(args.source, args.journal)],
                     sort_keys=True))


if __name__ == "__main__":
    main()
