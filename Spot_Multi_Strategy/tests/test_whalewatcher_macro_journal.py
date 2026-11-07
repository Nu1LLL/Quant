"""Synthetic WhaleWatcher macro provenance, without private live contents."""

import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from research.crypto_blackbox.whalewatcher_macro_journal import (
    capture_current, read_asof, upcoming_confirmed,
)


T0 = 2_000_000_000_000


def event(*, scheduled=T0 + 3_600_000, confirmed=True, withdrawn=False):
    return {"event": {
        "id": "official-cpi-example", "kind": "cpi", "source": "macro_calendar",
        "scheduled_at_ms": scheduled, "schedule_checked_at_ms": T0 - 60_000,
        "schedule_time_confirmed": confirmed,
        "schedule_provenance": "official_snapshot_2033_05_17",
        "schedule_url": "https://www.bls.gov/schedule/news_release/cpi.htm",
        "time_basis": "scheduled", "title": "unretained title",
    }, "schedule_withdrawn": withdrawn,
            "forecasts": [{"private_ai_probability": 0.9}],
            "markets": [{"private_market_odds": 0.7}],
            "recap": {"future_return": 0.5}}


class MacroJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "watcher" / "event_state" / "macro.json"
        self.source.parent.mkdir(parents=True)
        self.journal = root / "quant_macro.sqlite"

    def write(self, rows, *, generated=T0, duplicate=False):
        payload = {"schema_version": 1, "updated_at_ms": generated, "records": rows}
        raw = json.dumps(payload)
        if duplicate:
            raw = raw.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
        self.source.write_text(raw)

    def test_availability_revisions_and_redaction(self):
        self.write([event()])
        first = capture_current(self.source.parents[1], self.journal, observed_at_ms=T0)
        self.assertEqual((first.status, first.changed_events), ("fresh", 1))
        self.assertEqual(read_asof(self.journal, T0 - 1), ())
        visible = read_asof(self.journal, T0)
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0].kind, "cpi")
        self.assertEqual(visible[0].quant_available_at_ms, T0)
        self.assertEqual(len(upcoming_confirmed(self.journal, T0)), 1)
        self.assertNotIn("unretained", self.journal.read_bytes().decode("utf-8", "ignore"))
        self.assertNotIn("private_ai", self.journal.read_bytes().decode("utf-8", "ignore"))
        self.assertEqual(capture_current(self.source.parents[1], self.journal,
                                         observed_at_ms=T0 + 1).changed_events, 0)
        self.write([event(scheduled=T0 + 7_200_000)])
        capture_current(self.source.parents[1], self.journal, observed_at_ms=T0 + 2)
        self.assertEqual(read_asof(self.journal, T0 + 1)[0].scheduled_at_ms, T0 + 3_600_000)
        self.assertEqual(read_asof(self.journal, T0 + 2)[0].scheduled_at_ms, T0 + 7_200_000)
        self.write([event(scheduled=T0 + 7_200_000, withdrawn=True)])
        capture_current(self.source.parents[1], self.journal, observed_at_ms=T0 + 3)
        self.assertEqual(upcoming_confirmed(self.journal, T0 + 3), ())
        self.write([])
        capture_current(self.source.parents[1], self.journal, observed_at_ms=T0 + 4)
        self.assertEqual(read_asof(self.journal, T0 + 4), ())

    def test_stale_or_malformed_snapshot_does_not_erase_prior(self):
        self.write([event()])
        capture_current(self.source.parents[1], self.journal, observed_at_ms=T0)
        self.write([], generated=T0 - 700_000)
        status = capture_current(self.source.parents[1], self.journal, observed_at_ms=T0 + 1)
        self.assertEqual(status.status, "unavailable")
        self.assertEqual(len(read_asof(self.journal, T0 + 1)), 1)
        self.write([], duplicate=True)
        self.assertEqual(capture_current(self.source.parents[1], self.journal,
                                         observed_at_ms=T0 + 2).status, "unavailable")
        self.assertEqual(len(read_asof(self.journal, T0 + 2)), 1)

    def test_unconfirmed_schedule_and_immutable_history(self):
        self.write([event(confirmed=False)])
        capture_current(self.source.parents[1], self.journal, observed_at_ms=T0)
        self.assertEqual(len(read_asof(self.journal, T0)), 1)
        self.assertEqual(upcoming_confirmed(self.journal, T0), ())
        with sqlite3.connect(self.journal) as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("DELETE FROM changes")


if __name__ == "__main__":
    unittest.main()
