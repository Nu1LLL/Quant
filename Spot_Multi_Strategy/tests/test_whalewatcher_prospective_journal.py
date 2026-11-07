"""Synthetic point-in-time and redaction tests; no private live data copied."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from research.crypto_blackbox.whalewatcher_prospective_journal import (
    capture_current, counts_asof, poll_coverage, read_asof,
)


NOW = 1_800_000_000_000


def _event_snapshot(generated=NOW - 1_000, records=None):
    if records is None:
        records = [{
            "id": "private-row", "first_seen_at_ms": NOW - 30_000,
            "event": {
                "id": "bloomberg-private-id", "source": "bloomberg",
                "category": "crypto", "published_at_ms": NOW - 35_000,
                "url": "https://www.bloomberg.com/private?secret=never-copy",
                "title": "PRIVATE TITLE MARKER", "summary": "PRIVATE SUMMARY MARKER",
            },
            "study": {"post_event_return_pct": 150},
            "review": {"trade_action": "BUY NOW"},
        }]
    return {"schema_version": 1, "updated_at_ms": generated,
            "execution_eligible": False, "records": records}


def _claim_snapshot(generated=NOW - 1_000, observations=None):
    if observations is None:
        observations = [{
            "observation_id": "a" * 32, "source": "vergex",
            "fact_group_id": "chain:tx:" + "b" * 64,
            "available_at_ms": NOW - 60_000, "side": "short",
            "verification_status": "verified", "consensus_eligible": True,
            "execution_eligible": False,
            "summary": "PRIVATE CLAIM MARKER", "trade_action": "SELL NOW",
        }]
    return {
        "schema_version": 2, "generated_at_ms": generated,
        "signal_role": "external_intelligence_shadow",
        "execution_eligible": False, "operator_execution_eligible": False,
        "promotion_eligible": False, "manual_review_required": True,
        "performance_calibration_used": False,
        "symbols": {
            "BTC": {"manual_review_required": True, "execution_eligible": False,
                    "observations": observations},
            "ETH": {"manual_review_required": True, "execution_eligible": False,
                    "observations": []},
        },
    }


class ProspectiveJournalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.watcher = root / "watcher"
        self.events = self.watcher / "event_state" / "latest.json"
        self.claims = self.watcher / "external_intelligence_state" / "latest.json"
        self.events.parent.mkdir(parents=True)
        self.claims.parent.mkdir(parents=True)
        self.db = root / "journal.sqlite"
        self.now = NOW
        for module in (
            "research.crypto_blackbox.whalewatcher_event_shadow.time.time_ns",
            "research.crypto_blackbox.whalewatcher_shadow_adapter.time.time_ns",
        ):
            clock = patch(module, side_effect=lambda: self.now * 1_000_000)
            clock.start()
            self.addCleanup(clock.stop)

    def write(self, events, claims):
        self.events.write_text(json.dumps(events), encoding="utf-8")
        self.claims.write_text(json.dumps(claims), encoding="utf-8")

    def test_capture_is_redacted_deduplicated_and_asof_safe(self):
        self.write(_event_snapshot(), _claim_snapshot())
        first = capture_current(self.watcher, self.db)
        self.assertEqual([row.status for row in first], ["captured"] * 3)
        self.assertEqual(counts_asof(self.db, NOW - 1),
                         {"events": 0, "BTC_claims": 0, "ETH_claims": 0})
        self.assertEqual(counts_asof(self.db, NOW),
                         {"events": 1, "BTC_claims": 1, "ETH_claims": 0})
        at_first = read_asof(self.db, NOW)
        self.assertEqual(at_first.events[0].category, "crypto")
        self.assertEqual(at_first.events[0].quant_available_ms, NOW)
        self.assertEqual(at_first.btc_claims[0].asserted_side, "short")
        self.assertFalse(at_first.execution_eligible)
        self.assertTrue(all(not row.execution_eligible and not row.historical_oos_eligible
                            for row in first))
        self.assertEqual([row.status for row in capture_current(self.watcher, self.db)],
                         ["unchanged"] * 3)
        self.now = NOW + 1_000
        changed_body = _event_snapshot(self.now - 100)
        changed_body["records"][0]["event"]["title"] = "REVISED PRIVATE TITLE"
        self.write(changed_body, _claim_snapshot(self.now - 100))
        self.assertEqual([row.status for row in capture_current(self.watcher, self.db)],
                         ["unchanged"] * 3)
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM captures").fetchone()[0], 3)
            self.assertEqual(connection.execute("SELECT count(*) FROM source_polls").fetchone()[0], 3)
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute("DELETE FROM captures")
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute("DELETE FROM source_polls")
        raw = self.db.read_bytes()
        for forbidden in (b"PRIVATE", b"never-copy", b"BUY NOW", b"SELL NOW",
                          b"post_event_return_pct", b"trade_action"):
            self.assertNotIn(forbidden, raw)

        self.now = NOW + 2_000
        self.write(_event_snapshot(self.now - 1_000, []),
                   _claim_snapshot(self.now - 1_000, []))
        second = capture_current(self.watcher, self.db)
        self.assertEqual([row.status for row in second],
                         ["captured", "captured", "unchanged"])
        self.assertEqual(counts_asof(self.db, NOW + 1_000),
                         {"events": 1, "BTC_claims": 1, "ETH_claims": 0})
        self.assertEqual(counts_asof(self.db, self.now),
                         {"events": 0, "BTC_claims": 0, "ETH_claims": 0})
        coverage = poll_coverage(self.db, NOW - 1, self.now)
        self.assertEqual(coverage.polls, 4)
        self.assertEqual(coverage.unavailable_polls, 0)
        self.assertFalse(coverage.five_year_continuous_source)

    def test_event_correction_is_visible_only_after_new_capture(self):
        self.write(_event_snapshot(), _claim_snapshot())
        capture_current(self.watcher, self.db)
        self.now = NOW + 1_000
        corrected = _event_snapshot(self.now - 100)
        corrected["records"][0]["event"]["category"] = "regulation"
        self.write(corrected, _claim_snapshot(self.now - 100))
        self.assertEqual(capture_current(self.watcher, self.db)[0].status, "captured")
        self.assertEqual(read_asof(self.db, self.now - 1).events[0].category, "crypto")
        self.assertEqual(read_asof(self.db, self.now).events[0].category, "regulation")

    def test_unavailable_snapshots_do_not_create_journal(self):
        status = capture_current(self.watcher, self.db)
        self.assertEqual([row.status for row in status], ["unavailable"] * 3)
        self.assertFalse(self.db.exists())

    def test_existing_journal_records_unavailable_source_poll(self):
        self.write(_event_snapshot(), _claim_snapshot())
        capture_current(self.watcher, self.db)
        self.events.unlink()
        self.claims.unlink()
        self.now += 1_000
        status = capture_current(self.watcher, self.db)
        self.assertEqual([row.status for row in status], ["unavailable"] * 3)
        coverage = poll_coverage(self.db, NOW - 1, self.now)
        self.assertEqual(coverage.polls, 2)
        self.assertEqual(coverage.unavailable_polls, 1)
        self.assertFalse(coverage.five_year_continuous_source)

    def test_old_journal_without_polls_cannot_prove_coverage(self):
        with closing(sqlite3.connect(self.db)) as connection:
            connection.execute("CREATE TABLE legacy (id INTEGER)")
        coverage = poll_coverage(self.db, NOW, NOW + 1_000)
        self.assertEqual(coverage.polls, 0)
        self.assertFalse(coverage.five_year_continuous_source)

    def test_future_dated_source_event_is_not_visible_early(self):
        events = _event_snapshot()
        events["records"][0]["event"]["published_at_ms"] = NOW + 2_000
        self.write(events, _claim_snapshot())
        self.assertEqual(capture_current(self.watcher, self.db)[0].status, "captured")
        self.assertEqual(counts_asof(self.db, NOW)["events"], 0)
        self.assertEqual(counts_asof(self.db, NOW + 1_999)["events"], 0)
        self.assertEqual(counts_asof(self.db, NOW + 2_000)["events"], 1)
        self.assertEqual(read_asof(self.db, NOW).events, ())
        self.assertEqual(read_asof(self.db, NOW + 2_000).events[0].quant_available_ms,
                         NOW + 2_000)


if __name__ == "__main__":
    unittest.main()
