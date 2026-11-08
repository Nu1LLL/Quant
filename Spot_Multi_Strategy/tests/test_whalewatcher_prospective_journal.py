"""Synthetic point-in-time and redaction tests; no private live data copied."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from research.crypto_blackbox.whalewatcher_prospective_journal import (
    capture_current, counts_asof, poll_coverage, read_asof, read_events_asof,
)
from research.crypto_blackbox.whalewatcher_features import build_event_features
from research.crypto_blackbox.whalewatcher_news_features import build_news_features


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
        "source_health": {
            "schema_version": 1, "generated_at_ms": generated,
            "signal_role": "external_intelligence_shadow",
            "execution_eligible": False, "credentials_present_in_state": False,
            "status": "ready", "configured_source_count": 1,
            "ready_source_count": 1, "any_source_ready": True,
            "all_configured_sources_ready": True,
        },
        "readiness": {
            "status": "ready", "configured_source_count": 1,
            "ready_source_count": 1, "any_source_ready": True,
            "all_configured_sources_ready": True,
        },
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

    def test_features_use_quant_read_time_and_missing_poll_is_not_zero_news(self):
        self.write(_event_snapshot(), _claim_snapshot())
        capture_current(self.watcher, self.db)
        with patch("research.crypto_blackbox.whalewatcher_features.time.time_ns",
                   return_value=NOW * 1_000_000):
            features = build_event_features(self.db, NOW)
            missing = build_event_features(self.db, NOW - 1)
            with self.assertRaises(ValueError):
                build_event_features(self.db, NOW + 1)
        self.assertTrue(features.source_available)
        self.assertEqual(features.values["event_crypto_1h"], 1)
        self.assertEqual(features.values["event_crypto_24h"], 1)
        self.assertEqual(features.values["claim_BTC_short_24h"], 1)
        self.assertEqual(features.values["claim_ETH_short_24h"], 0)
        self.assertFalse(features.execution_eligible)
        self.assertFalse(features.historical_oos_eligible)
        self.assertFalse(missing.source_available)
        self.assertEqual(dict(missing.values), {})

    def test_features_wait_for_future_publication_and_expire_without_poll(self):
        future = _event_snapshot()
        future["records"][0]["event"]["published_at_ms"] = NOW + 2_000
        self.write(future, _claim_snapshot(observations=[]))
        capture_current(self.watcher, self.db)
        with patch("research.crypto_blackbox.whalewatcher_features.time.time_ns",
                   return_value=(NOW + 3 * 3_600_000) * 1_000_000):
            early = build_event_features(self.db, NOW)
            later = build_event_features(self.db, NOW + 2_000)
            stale = build_event_features(self.db, NOW + 3 * 3_600_000)
        self.assertEqual(early.values["event_crypto_1h"], 0)
        self.assertEqual(later.values["event_crypto_1h"], 1)
        self.assertFalse(stale.source_available)

    def test_news_remains_available_when_claim_feed_is_unavailable(self):
        self.write(_event_snapshot(), _claim_snapshot())
        capture_current(self.watcher, self.db)
        self.now += 1_000
        self.claims.unlink()
        statuses = capture_current(self.watcher, self.db)
        self.assertEqual([row.status for row in statuses],
                         ["unchanged", "unavailable", "unavailable"])
        self.assertEqual(poll_coverage(self.db, NOW - 1, self.now,
                                       stream="events").unavailable_polls, 0)
        self.assertEqual(poll_coverage(self.db, NOW - 1, self.now,
                                       stream="joint").unavailable_polls, 1)
        self.assertEqual(len(read_events_asof(self.db, self.now)), 1)
        with patch("research.crypto_blackbox.whalewatcher_news_features.time.time_ns",
                   return_value=self.now * 1_000_000):
            news = build_news_features(self.db, self.now)
        with patch("research.crypto_blackbox.whalewatcher_features.time.time_ns",
                   return_value=self.now * 1_000_000):
            combined = build_event_features(self.db, self.now)
        self.assertTrue(news.source_available)
        self.assertEqual(news.values["event_crypto_1h"], 1)
        self.assertFalse(news.execution_eligible)
        self.assertFalse(news.historical_oos_eligible)
        self.assertFalse(combined.source_available)
        self.assertEqual(dict(combined.values), {})
        self.now += 1_000
        self.events.unlink()
        capture_current(self.watcher, self.db)
        with patch("research.crypto_blackbox.whalewatcher_news_features.time.time_ns",
                   return_value=self.now * 1_000_000):
            missing = build_news_features(self.db, self.now)
        self.assertFalse(missing.source_available)
        self.assertEqual(dict(missing.values), {})

    def test_news_only_feature_waits_for_publication_and_ignores_old_import(self):
        future = _event_snapshot()
        future["records"][0]["event"]["published_at_ms"] = NOW + 2_000
        self.write(future, _claim_snapshot())
        self.claims.unlink()
        capture_current(self.watcher, self.db)
        with patch("research.crypto_blackbox.whalewatcher_news_features.time.time_ns",
                   return_value=(NOW + 2_000) * 1_000_000):
            early = build_news_features(self.db, NOW)
            later = build_news_features(self.db, NOW + 2_000)
        self.assertEqual(early.values["event_crypto_1h"], 0)
        self.assertEqual(later.values["event_crypto_1h"], 1)
        self.now += 3_000
        old = _event_snapshot(self.now - 100)
        old["records"][0]["event"]["published_at_ms"] = self.now - 3 * 24 * 3_600_000
        old["records"][0]["first_seen_at_ms"] = self.now - 24 * 3_600_000
        self.write(old, _claim_snapshot())
        self.claims.unlink()
        capture_current(self.watcher, self.db)
        with patch("research.crypto_blackbox.whalewatcher_news_features.time.time_ns",
                   return_value=self.now * 1_000_000):
            delayed = build_news_features(self.db, self.now)
        self.assertTrue(delayed.source_available)
        self.assertEqual(delayed.values["event_crypto_24h"], 0)

    def test_old_news_first_import_does_not_look_like_fresh_news(self):
        old = _event_snapshot()
        old["records"][0]["event"]["published_at_ms"] = NOW - 3 * 24 * 3_600_000
        old["records"][0]["first_seen_at_ms"] = NOW - 24 * 3_600_000
        old_claims = _claim_snapshot()
        old_claims["symbols"]["BTC"]["observations"][0]["available_at_ms"] = NOW - 3 * 24 * 3_600_000
        self.write(old, old_claims)
        capture_current(self.watcher, self.db)
        with patch("research.crypto_blackbox.whalewatcher_features.time.time_ns",
                   return_value=NOW * 1_000_000):
            features = build_event_features(self.db, NOW)
        self.assertTrue(features.source_available)
        self.assertEqual(features.values["event_crypto_1h"], 0)
        self.assertEqual(features.values["event_crypto_24h"], 0)
        self.assertEqual(features.values["claim_BTC_short_24h"], 0)


if __name__ == "__main__":
    unittest.main()
