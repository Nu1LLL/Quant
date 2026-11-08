"""Point-in-time and immutability checks for the prospective flow source."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from research.crypto_blackbox.hyperliquid_flow_prospective_journal import (
    ACTIVATION_MS, FIVE_MIN_MS, HOUR_MS, capture_once,
)


class FlowJournalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source.sqlite"
        self.journal = Path(self.tmp.name) / "journal.sqlite"
        with closing(sqlite3.connect(self.source)) as db, db:
            db.executescript("""
                CREATE TABLE market_flow_trades (
                    row_id INTEGER PRIMARY KEY, trade_id TEXT, timestamp_ms INTEGER,
                    side TEXT, notional_usd REAL, inserted_at_ms INTEGER,
                    venue TEXT, market_type TEXT, confirmation_status TEXT,
                    symbol TEXT);
                CREATE TABLE market_flow_source_health (
                    source TEXT PRIMARY KEY, status TEXT, healthy INTEGER,
                    connected INTEGER, last_event_at_ms INTEGER,
                    updated_at_ms INTEGER);
            """)

    def _seed_hour(self, start: int, now: int, *, malformed=False):
        with closing(sqlite3.connect(self.source)) as db, db:
            db.execute("""
                INSERT OR REPLACE INTO market_flow_source_health VALUES (?,?,?,?,?,?)
            """, ("hyperliquid_perpetual", "streaming", 1, 1,
                  start + HOUR_MS + 1000, now - 1000))
            for row_id, symbol, side, amount in (
                    (1, "BTC", "long", 100.10),
                    (2, "BTC", "short", 30.20),
                    (3, "ETH", "long", 40.30)):
                event_time = start + row_id * 1000
                db.execute("""
                    INSERT OR REPLACE INTO market_flow_trades VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (row_id, f"{event_time}:{row_id}", event_time, side, amount,
                      event_time + (600_000 if malformed and row_id == 2 else 10),
                      "hyperliquid", "perpetual", "executed", symbol))

    def test_timely_aggregates_and_append_only(self):
        start = ACTIVATION_MS + HOUR_MS
        now = start + HOUR_MS + FIVE_MIN_MS + 60_000
        self._seed_hour(start, now)
        captures = capture_once(self.source, self.journal, now_ms=now)
        self.assertEqual([(x.symbol, x.status, x.trade_count) for x in captures],
                         [("BTC", "timely", 2), ("ETH", "timely", 1)])
        self.assertEqual((captures[0].buy_cents, captures[0].sell_cents),
                         (10010, 3020))
        self.assertEqual(capture_once(self.source, self.journal, now_ms=now), ())
        with closing(sqlite3.connect(self.journal)) as db, db:
            with self.assertRaisesRegex(sqlite3.DatabaseError, "append-only"):
                db.execute("UPDATE hourly_flow SET status='late'")

    def test_future_activation_and_no_backfill(self):
        self.assertEqual(capture_once(self.source, self.journal,
                                      now_ms=ACTIVATION_MS + HOUR_MS), ())
        first = ACTIVATION_MS + HOUR_MS
        first_now = first + HOUR_MS + FIVE_MIN_MS + 1
        self._seed_hour(first, first_now)
        capture_once(self.source, self.journal, now_ms=first_now)
        later = first + 2 * HOUR_MS
        later_now = later + HOUR_MS + FIVE_MIN_MS + 1
        self._seed_hour(later, later_now)
        capture_once(self.source, self.journal, now_ms=later_now)
        with closing(sqlite3.connect(self.journal)) as db, db:
            windows = [row[0] for row in db.execute(
                "SELECT DISTINCT window_start_ms FROM hourly_flow ORDER BY 1")]
        self.assertEqual(windows, [first, later])

    def test_late_and_invalid_source_cannot_be_timely(self):
        start = ACTIVATION_MS + HOUR_MS
        late_now = start + HOUR_MS + 20 * 60_000
        self._seed_hour(start, late_now)
        late = capture_once(self.source, self.journal, now_ms=late_now)
        self.assertEqual([item.status for item in late], ["late", "late"])
        second_journal = Path(self.tmp.name) / "invalid.sqlite"
        self._seed_hour(start, late_now, malformed=True)
        invalid = capture_once(self.source, second_journal, now_ms=late_now)
        self.assertEqual([item.status for item in invalid],
                         ["invalid", "invalid"])

    def test_trade_not_visible_at_observation_is_rejected(self):
        start = ACTIVATION_MS + HOUR_MS
        now = start + HOUR_MS + FIVE_MIN_MS + 60_000
        self._seed_hour(start, now)
        with closing(sqlite3.connect(self.source)) as db, db:
            db.execute("UPDATE market_flow_trades SET inserted_at_ms=? WHERE row_id=2",
                       (now + 1000,))
        captures = capture_once(self.source, self.journal, now_ms=now)
        self.assertEqual([item.status for item in captures], ["invalid", "invalid"])

    def test_old_hyperliquid_tid_only_format_is_not_eligible(self):
        start = ACTIVATION_MS + HOUR_MS
        now = start + HOUR_MS + FIVE_MIN_MS + 60_000
        self._seed_hour(start, now)
        with closing(sqlite3.connect(self.source)) as db, db:
            db.execute("UPDATE market_flow_trades SET trade_id='2' WHERE row_id=2")
        captures = capture_once(self.source, self.journal, now_ms=now)
        self.assertEqual([item.status for item in captures], ["invalid", "invalid"])


if __name__ == "__main__":
    unittest.main()
