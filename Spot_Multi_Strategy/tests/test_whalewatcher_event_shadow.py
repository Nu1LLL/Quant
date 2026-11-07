"""Only synthetic WhaleWatcher journals; no local private content is copied."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.crypto_blackbox.whalewatcher_event_shadow import (
    read_whalewatcher_event_shadow,
)


NOW = 1_800_000_000_000


def journal():
    return {
        "schema_version": 1, "updated_at_ms": NOW - 1_000,
        "execution_eligible": False,
        "records": [{
            "id": "private-row", "first_seen_at_ms": NOW - 30_000,
            "event": {
                "id": "bloomberg-private-id", "source": "bloomberg",
                "category": "crypto", "published_at_ms": NOW - 35_000,
                "url": "https://www.bloomberg.com/article?private=secret-marker",
                "title": "PRIVATE TITLE MARKER", "summary": "PRIVATE SUMMARY MARKER",
            },
            "study": {"post_event_return_pct": 150},
            "review": {"trade_action": "BUY NOW"},
            "account": {"wallet": "PRIVATE WALLET MARKER"},
        }],
    }


class EventShadowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "latest.json"
        clock = patch(
            "research.crypto_blackbox.whalewatcher_event_shadow.time.time_ns",
            return_value=NOW * 1_000_000,
        )
        clock.start()
        self.addCleanup(clock.stop)

    def read(self, data=None):
        if data is not None:
            self.path.write_text(json.dumps(data), encoding="utf-8")
        return read_whalewatcher_event_shadow(self.path)

    def test_fresh_event_has_quant_read_time_and_no_private_or_trade_content(self):
        imported = self.read(journal())
        self.assertEqual(imported.status, "fresh")
        self.assertEqual(len(imported.events), 1)
        event = imported.events[0]
        self.assertEqual(event.quant_available_ms, NOW)
        self.assertEqual(event.watcher_first_seen_ms, NOW - 30_000)
        self.assertFalse(event.execution_eligible)
        self.assertFalse(event.historical_oos_eligible)
        serialized = json.dumps(asdict(imported))
        for forbidden in ("PRIVATE", "secret-marker", "BUY NOW", "post_event_return_pct",
                          "account", "title", "summary", "wallet", "trade_action"):
            self.assertNotIn(forbidden, serialized)

    def test_missing_stale_future_duplicate_and_untrusted_urls_fail_closed(self):
        self.assertEqual(self.read().status, "unavailable")
        for change in ("stale", "future", "duplicate", "url", "first_seen"):
            data = journal()
            if change == "stale":
                data["updated_at_ms"] = NOW - 600_001
            elif change == "future":
                data["updated_at_ms"] = NOW + 1
            elif change == "duplicate":
                data["records"].append(data["records"][0].copy())
            elif change == "url":
                data["records"][0]["event"]["url"] = "file:///private/secrets"
            else:
                data["records"][0]["first_seen_at_ms"] = NOW + 1
            with self.subTest(change=change):
                self.assertEqual(self.read(data).status, "unavailable")

    def test_non_source_rows_are_not_imported_as_market_events(self):
        data = journal()
        data["records"][0]["event"]["source"] = "verified_calendar"
        result = self.read(data)
        self.assertEqual(result.status, "fresh_no_events")
        self.assertEqual(result.events, ())


if __name__ == "__main__":
    unittest.main()
