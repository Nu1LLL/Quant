"""Compact, prospective Binance spot aggressor-flow observations; never orders.

Only completed UTC hours are stored. A websocket interruption or aggregate-trade
ID gap makes the affected hour ineligible. No past trade is silently backfilled.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from pathlib import Path
import sqlite3
import ssl
import time


HOUR_MS = 3_600_000
ACTIVATION_MS = 1_794_182_400_000  # 2026-11-09 00:00 UTC
GRACE_MS = 60_000
MAX_EVENT_LAG_MS = 10_000
MAX_DB_BYTES = 96 * 1024 * 1024
DB_HEADROOM_BYTES = 4 * 1024 * 1024
SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "SOLUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT", "DOTUSDT", "AVAXUSDT",
)
JOURNAL = Path.home() / "Library/Application Support/MiniMedallion/binance_spot_flow_prospective.sqlite"
URL = ("wss://data-stream.binance.vision/stream?streams="
       + "/".join(symbol.lower() + "@aggTrade" for symbol in SYMBOLS))


@dataclass(frozen=True)
class Trade:
    symbol: str
    agg_id: int
    trade_ms: int
    event_ms: int
    buy_micros: int
    sell_micros: int


@dataclass
class Bucket:
    count: int = 0
    buys: int = 0
    sells: int = 0
    buy_micros: int = 0
    sell_micros: int = 0
    first_id: int | None = None
    last_id: int | None = None
    first_received_ms: int | None = None
    last_received_ms: int | None = None


def parse_trade(raw: str, received_ms: int) -> Trade:
    """Validate only documented aggregate trade fields, with bounded latency."""
    if len(raw) > 4096:
        raise ValueError("oversized websocket event")
    try:
        outer = json.loads(raw)
        data = outer["data"]
        symbol = data["s"]
        if (symbol not in SYMBOLS or outer["stream"] != symbol.lower() + "@aggTrade"
                or data["e"] != "aggTrade" or type(data["m"]) is not bool):
            raise ValueError("unexpected aggregate-trade stream")
        agg_id, trade_ms, event_ms = (data[key] for key in ("a", "T", "E"))
        if any(type(x) is not int or x < 0 for x in (agg_id, trade_ms, event_ms)):
            raise ValueError("invalid aggregate-trade ID or timestamp")
        if (trade_ms > event_ms + 1000 or event_ms > received_ms + 2000
                or received_ms - trade_ms > MAX_EVENT_LAG_MS):
            raise ValueError("late or future aggregate trade")
        price, quantity = Decimal(data["p"]), Decimal(data["q"])
        if (not price.is_finite() or not quantity.is_finite()
                or price <= 0 or quantity <= 0):
            raise ValueError("invalid price or quantity")
        notional = int((price * quantity * 1_000_000).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP))
        if notional <= 0 or notional >= 2**63:
            raise ValueError("notional outside storage range")
        return Trade(symbol, agg_id, trade_ms, event_ms,
                     0 if data["m"] else notional,
                     notional if data["m"] else 0)
    except (KeyError, TypeError, InvalidOperation, OverflowError) as exc:
        raise ValueError("malformed aggregate-trade event") from exc


def _db_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in (path, Path(str(path) + "-wal")) if p.exists())


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=5)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS source_contract (id INTEGER PRIMARY KEY CHECK(id=1), contract TEXT NOT NULL)")
    contract = json.dumps({"activation_ms": ACTIVATION_MS, "symbols": SYMBOLS,
                           "schema": 1, "source": "binance_spot_aggTrade"},
                          sort_keys=True, separators=(",", ":"))
    db.execute("INSERT OR IGNORE INTO source_contract VALUES (1,?)", (contract,))
    if db.execute("SELECT contract FROM source_contract WHERE id=1").fetchone()[0] != contract:
        db.close()
        raise RuntimeError("journal source contract differs from frozen collector")
    db.execute("""
        CREATE TABLE IF NOT EXISTS hourly_flow (
            symbol TEXT NOT NULL,
            window_start_ms INTEGER NOT NULL,
            recorded_at_ms INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('observed','empty','invalid','unavailable')),
            count INTEGER NOT NULL CHECK(count>=0),
            buys INTEGER NOT NULL CHECK(buys>=0),
            sells INTEGER NOT NULL CHECK(sells>=0),
            buy_micros INTEGER NOT NULL CHECK(buy_micros>=0),
            sell_micros INTEGER NOT NULL CHECK(sell_micros>=0),
            first_id INTEGER,
            last_id INTEGER,
            first_received_ms INTEGER,
            last_received_ms INTEGER,
            PRIMARY KEY(symbol,window_start_ms),
            CHECK(window_start_ms>=1794182400000),
            CHECK(count=buys+sells)
        ) WITHOUT ROWID
    """)
    db.execute("""
        CREATE TRIGGER IF NOT EXISTS hourly_flow_no_update
        BEFORE UPDATE ON hourly_flow BEGIN SELECT RAISE(ABORT,'append-only'); END
    """)
    db.execute("""
        CREATE TRIGGER IF NOT EXISTS hourly_flow_no_delete
        BEFORE DELETE ON hourly_flow BEGIN SELECT RAISE(ABORT,'append-only'); END
    """)
    db.commit()
    return db


class FlowCollector:
    """Fail-closed hourly accumulator. State is only held in memory until sealed."""

    def __init__(self, db: sqlite3.Connection, path: Path):
        self.db = db
        self.path = path
        latest = db.execute("SELECT MAX(window_start_ms) FROM hourly_flow").fetchone()[0]
        self.next_start = ACTIVATION_MS if latest is None else max(
            ACTIVATION_MS, int(latest) + HOUR_MS)
        self.ready_from: int | None = None
        self.last_id: dict[str, int] = {}
        self.buckets: dict[tuple[str, int], Bucket] = {}
        self.invalid: set[tuple[str, int]] = set()
        self.first_observed_ms: int | None = None

    def connected(self, at_ms: int) -> None:
        # The partial connection hour is never called a continuous observation.
        self.ready_from = ((at_ms // HOUR_MS) + 1) * HOUR_MS
        self.first_observed_ms = at_ms
        self.last_id.clear()

    def disconnected(self, at_ms: int) -> None:
        # Any open hour touched by the interruption is not source-complete.
        start = (at_ms // HOUR_MS) * HOUR_MS
        for symbol in SYMBOLS:
            self.invalid.add((symbol, start))
        self.ready_from = None
        self.last_id.clear()

    def invalidate_current(self, at_ms: int) -> None:
        start = (at_ms // HOUR_MS) * HOUR_MS
        for symbol in SYMBOLS:
            self.invalid.add((symbol, start))
            # A malformed message just after the boundary may belong to the
            # preceding hour; it is still unsealed during the grace period.
            if at_ms - start <= GRACE_MS and start - HOUR_MS >= self.next_start:
                self.invalid.add((symbol, start - HOUR_MS))

    def ingest(self, trade: Trade, received_ms: int) -> None:
        start = (trade.trade_ms // HOUR_MS) * HOUR_MS
        if start < ACTIVATION_MS:
            self.last_id[trade.symbol] = trade.agg_id
            return
        if start < self.next_start:
            raise RuntimeError("late aggregate trade for a sealed hour")
        key = (trade.symbol, start)
        previous = self.last_id.get(trade.symbol)
        if previous is not None and trade.agg_id != previous + 1:
            self.invalid.add(key)
            # If a gap straddles an unsealed hour boundary, invalidate both.
            prior = (trade.symbol, start - HOUR_MS)
            if prior[1] >= self.next_start:
                self.invalid.add(prior)
        self.last_id[trade.symbol] = trade.agg_id
        if received_ms - trade.trade_ms > MAX_EVENT_LAG_MS:
            self.invalid.add(key)
        bucket = self.buckets.setdefault(key, Bucket())
        bucket.count += 1
        bucket.buys += bool(trade.buy_micros)
        bucket.sells += bool(trade.sell_micros)
        bucket.buy_micros += trade.buy_micros
        bucket.sell_micros += trade.sell_micros
        bucket.first_id = trade.agg_id if bucket.first_id is None else bucket.first_id
        bucket.last_id = trade.agg_id
        bucket.first_received_ms = (received_ms if bucket.first_received_ms is None
                                    else bucket.first_received_ms)
        bucket.last_received_ms = received_ms

    def seal_due(self, now_ms: int) -> int:
        count = 0
        while self.next_start + HOUR_MS + GRACE_MS <= now_ms:
            if _db_bytes(self.path) >= MAX_DB_BYTES - DB_HEADROOM_BYTES:
                raise RuntimeError("prospective Binance journal reached 96 MiB safety threshold")
            start = self.next_start
            rows = []
            for symbol in SYMBOLS:
                key = (symbol, start)
                bucket = self.buckets.pop(key, Bucket())
                status = ("invalid" if key in self.invalid else
                          "unavailable" if self.ready_from is None or start < self.ready_from else
                          "empty" if bucket.count == 0 else "observed")
                rows.append((symbol, start, now_ms, status, bucket.count,
                             bucket.buys, bucket.sells, bucket.buy_micros,
                             bucket.sell_micros, bucket.first_id, bucket.last_id,
                             bucket.first_received_ms, bucket.last_received_ms))
                self.invalid.discard(key)
            with self.db:
                self.db.executemany("INSERT INTO hourly_flow VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self.next_start += HOUR_MS
            count += len(rows)
        return count


async def run(path: Path = JOURNAL) -> None:
    import certifi
    import websockets

    # No historic backfill: start listening shortly before the fixed cutover.
    while time.time_ns() // 1_000_000 < ACTIVATION_MS - 600_000:
        await asyncio.sleep(min(300, max(1, (ACTIVATION_MS - 600_000
                                        - time.time_ns() // 1_000_000) / 1000)))
    with closing(_connect(path)) as db:
        collector = FlowCollector(db, path)
        tls = ssl.create_default_context(cafile=certifi.where())
        backoff = 1
        while True:
            try:
                async with websockets.connect(URL, ssl=tls, open_timeout=20,
                                              max_size=4096, ping_interval=20) as ws:
                    collector.connected(time.time_ns() // 1_000_000)
                    backoff = 1
                    while True:
                        now = time.time_ns() // 1_000_000
                        collector.seal_due(now)
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5)
                            received = time.time_ns() // 1_000_000
                            if isinstance(raw, bytes):
                                raise ValueError("binary event on JSON stream")
                            outer = json.loads(raw)
                            if outer.get("data", {}).get("e") == "serverShutdown":
                                break
                            collector.ingest(parse_trade(raw, received), received)
                        except asyncio.TimeoutError:
                            continue
                        except (ValueError, KeyError, TypeError, AttributeError):
                            collector.invalidate_current(time.time_ns() // 1_000_000)
            except (OSError, TimeoutError, ssl.SSLError, websockets.exceptions.WebSocketException):
                pass
            collector.disconnected(time.time_ns() // 1_000_000)
            collector.seal_due(time.time_ns() // 1_000_000)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=JOURNAL)
    args = parser.parse_args()
    asyncio.run(run(args.journal))


if __name__ == "__main__":
    main()
