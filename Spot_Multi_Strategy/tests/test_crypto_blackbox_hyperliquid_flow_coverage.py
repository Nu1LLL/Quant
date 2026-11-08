"""Synthetic coverage audit; no private source rows, prices, or outcomes."""

from contextlib import closing
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research.crypto_blackbox.hyperliquid_flow_coverage import (
    FlowCoverageError, audit_coverage,
)
from research.crypto_blackbox.hyperliquid_flow_prospective_journal import (
    ACTIVATION_MS, FIVE_MIN_MS, HOUR_MS, _journal_connection,
)


class FlowCoverageTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.journal = Path(self.folder.name) / "future.sqlite"

    def insert(self, start: int, symbol: str, *, status: str = "timely",
               delay_ms: int = FIVE_MIN_MS + 1) -> int:
        end = start + HOUR_MS
        observed = end + delay_ms
        count = 0 if status in ("empty", "unavailable", "invalid") else 1
        with closing(_journal_connection(self.journal)) as db, db:
            db.execute("INSERT INTO hourly_flow VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (symbol, start, end, observed, status, observed - 1000,
                        count, count, 0, count * 100, 0,
                        sha256(b"synthetic").hexdigest()))
        return observed

    def test_pre_activation_and_missing_due_hour_do_not_create_journal(self):
        before = audit_coverage(self.journal, as_of_ms=ACTIVATION_MS + HOUR_MS)
        self.assertEqual(before["status"], "awaiting_first_complete_hour")
        self.assertEqual(before["expected_hours"], 0)
        due = audit_coverage(self.journal,
                             as_of_ms=ACTIVATION_MS + HOUR_MS + FIVE_MIN_MS)
        self.assertEqual((due["status"], due["expected_hours"], due["missing_rows"]),
                         ("journal_missing", 1, 2))
        self.assertFalse(due["five_year_independent_oos_proven"])
        self.assertFalse(self.journal.exists())

    def test_timely_rows_and_permanent_missing_hour(self):
        self.insert(ACTIVATION_MS, "BTC")
        now = self.insert(ACTIVATION_MS, "ETH")
        first = audit_coverage(self.journal, as_of_ms=now)
        self.assertEqual((first["expected_hours"], first["timely_hours_both_symbols"],
                          first["bad_hours"]), (1, 1, 0))
        self.assertFalse(first["first_five_year_source_coverage_complete"])
        next_due = ACTIVATION_MS + 2 * HOUR_MS + FIVE_MIN_MS
        gap = audit_coverage(self.journal, as_of_ms=next_due)
        self.assertEqual((gap["expected_hours"], gap["missing_rows"],
                          gap["bad_hours"], gap["max_consecutive_bad_hours"]),
                         (2, 2, 1, 1))
        self.assertEqual(gap["first_bad_windows_ms"],
                         [ACTIVATION_MS + HOUR_MS])
        self.assertFalse(gap["five_year_independent_oos_proven"])

    def test_late_and_invalid_claimed_timely_are_not_accepted(self):
        self.insert(ACTIVATION_MS, "BTC", status="late", delay_ms=16 * 60_000)
        as_of = self.insert(ACTIVATION_MS, "ETH", status="late",
                            delay_ms=16 * 60_000)
        before_observation = audit_coverage(
            self.journal, as_of_ms=ACTIVATION_MS + HOUR_MS + FIVE_MIN_MS)
        self.assertEqual((before_observation["bad_hours"],
                          before_observation["missing_rows"]), (1, 2))
        late = audit_coverage(self.journal, as_of_ms=as_of)
        self.assertEqual((late["bad_hours"], late["timely_hours_both_symbols"]),
                         (1, 0))
        forged = Path(self.folder.name) / "forged.sqlite"
        self.journal = forged
        self.insert(ACTIVATION_MS, "BTC", status="timely",
                    delay_ms=16 * 60_000)
        with self.assertRaisesRegex(FlowCoverageError, "captured late"):
            audit_coverage(self.journal, as_of_ms=as_of)


if __name__ == "__main__":
    unittest.main()
