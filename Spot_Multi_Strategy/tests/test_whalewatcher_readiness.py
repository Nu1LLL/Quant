"""Readiness checks use synthetic timestamps and temporary journals only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

from research.crypto_blackbox.whalewatcher_readiness import (
    MAX_POLL_GAP_MS, _five_calendar_years_elapsed, _stream, assess,
)


def ms(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


class WhaleWatcherReadinessTests(unittest.TestCase):
    def test_five_calendar_years_including_leap_day(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        anniversary = start.replace(year=2031)
        self.assertFalse(_five_calendar_years_elapsed(
            ms(start), ms(anniversary - timedelta(milliseconds=1))))
        self.assertTrue(_five_calendar_years_elapsed(ms(start), ms(anniversary)))

    def test_continuous_five_year_stream_requires_current_poll(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start.replace(year=2031)
        interval = timedelta(hours=2)
        rows = []
        point = start
        while point <= end:
            rows.append((ms(point), "fresh"))
            point += interval
        ready = _stream("macro_schedule", rows, ms(end))
        self.assertTrue(ready.five_year_continuous_source)
        self.assertEqual(ready.latest_continuous_start_ms, ms(start))
        stale = _stream("macro_schedule", rows, ms(end) + MAX_POLL_GAP_MS + 1)
        self.assertFalse(stale.five_year_continuous_source)
        self.assertEqual(stale.state, "stale")

    def test_gap_and_unavailable_reset_continuous_start(self):
        base = ms(datetime(2026, 11, 8, tzinfo=timezone.utc))
        hour = 3_600_000
        rows = [(base, "fresh"), (base + hour, "fresh"),
                (base + 4 * hour, "fresh"), (base + 5 * hour, "unavailable"),
                (base + 6 * hour, "fresh")]
        status = _stream("macro_schedule", rows, base + 6 * hour)
        self.assertEqual(status.latest_continuous_start_ms, base + 6 * hour)
        self.assertEqual(status.latest_continuous_polls, 1)
        self.assertEqual(status.breaks_total, 2)

    def test_joint_audit_is_read_only_and_requires_both_sources(self):
        base = ms(datetime(2026, 11, 8, tzinfo=timezone.utc))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = assess(root, base)
            self.assertEqual(missing.events_claims.state, "missing")
            self.assertEqual(missing.news_events.state, "missing")
            self.assertEqual(list(root.iterdir()), [])
            with sqlite3.connect(root / "whalewatcher_shadow.sqlite") as db:
                db.execute("CREATE TABLE source_polls (id INTEGER PRIMARY KEY, "
                           "observed_at_ms INTEGER, events_status TEXT, "
                           "btc_status TEXT, eth_status TEXT)")
                db.execute("INSERT INTO source_polls VALUES (1,?,?,?,?)",
                           (base, "fresh", "fresh_no_claims", "fresh_no_claims"))
            with sqlite3.connect(root / "whalewatcher_macro.sqlite") as db:
                db.execute("CREATE TABLE polls (id INTEGER PRIMARY KEY, "
                           "observed_at_ms INTEGER, status TEXT)")
                db.execute("INSERT INTO polls VALUES (1,?,?)",
                           (base + 1_000, "fresh"))
            status = assess(root, base + 2_000)
            self.assertEqual(status.joint_start_ms, base + 1_000)
            self.assertEqual(status.news_events.state, "fresh")
            self.assertFalse(status.five_year_continuous_source)
            self.assertFalse(status.independent_trading_oos_proven)
            self.assertFalse(status.execution_eligible)

    def test_news_continuity_is_separate_from_claim_outage(self):
        base = ms(datetime(2026, 11, 8, tzinfo=timezone.utc))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with sqlite3.connect(root / "whalewatcher_shadow.sqlite") as db:
                db.execute("CREATE TABLE source_polls (id INTEGER PRIMARY KEY, "
                           "observed_at_ms INTEGER, events_status TEXT, "
                           "btc_status TEXT, eth_status TEXT)")
                db.execute("INSERT INTO source_polls VALUES (1,?,?,?,?)",
                           (base, "fresh", "unavailable", "unavailable"))
            with sqlite3.connect(root / "whalewatcher_macro.sqlite") as db:
                db.execute("CREATE TABLE polls (id INTEGER PRIMARY KEY, "
                           "observed_at_ms INTEGER, status TEXT)")
                db.execute("INSERT INTO polls VALUES (1,?,?)", (base, "fresh"))
            result = assess(root, base + 1_000)
            self.assertEqual(result.news_events.state, "fresh")
            self.assertEqual(result.events_claims.state, "unavailable")
            self.assertEqual(result.joint_span_ms, 0)
            self.assertFalse(result.five_year_continuous_source)


if __name__ == "__main__":
    unittest.main()
