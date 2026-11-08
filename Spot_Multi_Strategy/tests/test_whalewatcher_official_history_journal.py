"""Synthetic provenance, first-seen, revision and append-only checks."""

import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from research.crypto_blackbox.whalewatcher_official_history_journal import (
    capture_current, read_asof,
)


OBSERVED = 1_800_000_000_000
RELEASED = 1_790_000_000_000


def event(*, url="https://www.bls.gov/news.release/example.htm",
          released=RELEASED):
    return {
        "id": "nfp-official-archive-example", "kind": "nfp",
        "source": "verified_calendar", "publisher": "bls",
        "category": "inflation_employment", "importance": "high",
        "published_at_ms": released, "occurred_at_ms": released,
        "verified_at_ms": released + 1000, "timestamp_verified": True,
        "time_basis": "occurred",
        "timestamp_basis": "official_archive_embargo_header_snapshot",
        "release_time_evidence": "synthetic embargo header",
        "release_timezone": "EDT", "release_utc_offset_minutes": -240,
        "url": url, "title": "must not be copied", "summary": "must not be copied",
    }


def source(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "events": rows,
                                "other_untrusted_payload": "never copied"}))


class OfficialHistoryJournalTests(unittest.TestCase):
    def test_first_seen_is_quant_capture_not_old_release(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, journal = root / "source.json", root / "journal.sqlite"
            source(upstream, [event()])
            first = capture_current(upstream, journal, observed_at_ms=OBSERVED)
            self.assertEqual((first.status, first.active_events, first.changes),
                             ("fresh", 1, 1))
            self.assertEqual(read_asof(journal, OBSERVED - 1), ())
            rows = read_asof(journal, OBSERVED)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].released_at_ms, RELEASED)
            self.assertEqual(rows[0].quant_first_seen_at_ms, OBSERVED)
            self.assertFalse(rows[0].historical_oos_eligible)
            self.assertFalse(rows[0].execution_eligible)
            self.assertNotIn("must not be copied", journal.read_bytes().decode("latin1"))

    def test_revisions_and_removal_do_not_rewrite_history(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, journal = root / "source.json", root / "journal.sqlite"
            source(upstream, [event()])
            capture_current(upstream, journal, observed_at_ms=OBSERVED)
            source(upstream, [event(released=RELEASED + 86_400_000)])
            revised = capture_current(upstream, journal,
                                      observed_at_ms=OBSERVED + 1000)
            self.assertEqual(revised.changes, 1)
            self.assertEqual(read_asof(journal, OBSERVED)[0].released_at_ms,
                             RELEASED)
            latest = read_asof(journal, OBSERVED + 1000)[0]
            self.assertEqual(latest.released_at_ms, RELEASED + 86_400_000)
            self.assertEqual(latest.quant_first_seen_at_ms, OBSERVED)
            self.assertEqual(latest.quant_version_seen_at_ms, OBSERVED + 1000)
            source(upstream, [])
            removed = capture_current(upstream, journal,
                                      observed_at_ms=OBSERVED + 2000)
            self.assertEqual(removed.changes, 1)
            self.assertEqual(read_asof(journal, OBSERVED + 2000), ())
            self.assertEqual(len(read_asof(journal, OBSERVED)), 1)
            source(upstream, [event()])
            capture_current(upstream, journal, observed_at_ms=OBSERVED + 3000)
            readded = read_asof(journal, OBSERVED + 3000)[0]
            self.assertEqual(readded.quant_first_seen_at_ms, OBSERVED)
            self.assertEqual(readded.quant_version_seen_at_ms, OBSERVED + 3000)
            with sqlite3.connect(journal) as db:
                with self.assertRaises(sqlite3.DatabaseError):
                    db.execute("DELETE FROM polls")

    def test_malformed_source_records_unavailable_without_erasing_asof(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, journal = root / "source.json", root / "journal.sqlite"
            source(upstream, [event()])
            capture_current(upstream, journal, observed_at_ms=OBSERVED)
            source(upstream, [event(url="https://evil.example/report")])
            unavailable = capture_current(upstream, journal,
                                          observed_at_ms=OBSERVED + 1000)
            self.assertEqual(unavailable.status, "unavailable")
            self.assertEqual(unavailable.changes, 0)
            self.assertEqual(len(read_asof(journal, OBSERVED + 1000)), 1)


if __name__ == "__main__":
    unittest.main()
