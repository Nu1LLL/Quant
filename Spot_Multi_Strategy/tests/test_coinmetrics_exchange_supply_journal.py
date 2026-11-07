"""Synthetic point-in-time observation and revision tests (no network)."""

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sqlite3
import unittest

from research.crypto_blackbox.coinmetrics_exchange_supply_journal import (
    JournalError, capture_current, read_asof, source_coverage,
)


def _response(*values: tuple[str, str]) -> bytes:
    return json.dumps({"data": [
        {"asset": "btc", "time": day + "T00:00:00.000000000Z", "SplyExNtv": amount}
        for day, amount in values
    ]}, separators=(",", ":")).encode("utf-8")


def _utc(day: int, hour: int = 1, second: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, 0, second, tzinfo=timezone.utc)


class ExchangeSupplyJournalTests(unittest.TestCase):
    def test_future_information_is_hidden_and_revisions_are_append_only(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "space journal.sqlite"
            initial = _response(("2026-11-01", "100"), ("2026-11-02", "200"))
            first = capture_current(path, fetcher=lambda _: initial,
                                    poll_time=_utc(8), received_at=_utc(8, second=1))
            self.assertEqual((first.rows, first.new_versions), (2, 2))
            self.assertEqual(path.stat().st_mode & 0o077, 0)
            self.assertIsNone(read_asof(path, date(2026, 11, 1), _utc(8, second=0)))
            seen = read_asof(path, date(2026, 11, 1), _utc(8, second=2))
            self.assertEqual(str(seen.value), "100")

            repeat = capture_current(path, fetcher=lambda _: initial,
                                     poll_time=_utc(9), received_at=_utc(9, second=1))
            self.assertEqual(repeat.new_versions, 0)
            revised = _response(("2026-11-01", "180"), ("2026-11-02", "200"))
            third = capture_current(path, fetcher=lambda _: revised,
                                    poll_time=_utc(10), received_at=_utc(10, second=1))
            self.assertEqual(third.new_versions, 1)
            self.assertEqual(str(read_asof(path, date(2026, 11, 1), _utc(9, 12)).value), "100")
            self.assertEqual(str(read_asof(path, date(2026, 11, 1), _utc(10, 12)).value), "180")

    def test_rejects_wrong_asset_and_early_decision_date(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "journal.sqlite"
            bad = b'{"data":[{"asset":"eth","time":"2026-11-01T00:00:00Z","SplyExNtv":"100"}]}'
            with self.assertRaisesRegex(JournalError, "wrong asset"):
                capture_current(path, fetcher=lambda _: bad,
                                poll_time=_utc(8), received_at=_utc(8, second=1))
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(JournalError, "observed date must precede"):
                read_asof(path, date(2026, 11, 8), _utc(8))

    def test_source_coverage_does_not_backfill_old_dates_or_missing_poll_days(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "journal.sqlite"
            self.assertEqual(source_coverage(path, asof=_utc(8)).polls, 0)
            for poll_day, values in (
                (8, (("2026-11-01", "100"),)),
                (9, (("2026-11-08", "101"),)),
                (11, (("2026-11-09", "102"),)),
            ):
                capture_current(path, fetcher=lambda _, items=values: _response(*items),
                                poll_time=_utc(poll_day), received_at=_utc(poll_day, second=1))
            coverage = source_coverage(path, asof=_utc(11, hour=7))
            self.assertEqual(coverage.polls, 3)
            self.assertEqual(coverage.missing_poll_days, 1)
            self.assertEqual(coverage.missing_or_late_observation_days, 0)
            self.assertEqual(coverage.largest_poll_gap_hours, 48)
            self.assertFalse(coverage.five_year_continuous_source)
            later = source_coverage(path, asof=_utc(13, hour=7))
            self.assertEqual(later.missing_poll_days, 3)
            self.assertGreater(later.missing_or_late_observation_days, 0)

    def test_records_cannot_be_updated_or_deleted(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "journal.sqlite"
            capture_current(path, fetcher=lambda _: _response(("2026-11-01", "100")),
                            poll_time=_utc(8), received_at=_utc(8, second=1))
            with sqlite3.connect(path) as db:
                for statement in (
                    "UPDATE polls SET row_count=0",
                    "DELETE FROM polls",
                    "UPDATE versions SET value_text='0'",
                    "DELETE FROM versions",
                ):
                    with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                        db.execute(statement)

    def test_late_or_missing_daily_value_is_not_counted_as_continuous(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "journal.sqlite"
            for poll_day, values in (
                (8, (("2026-11-01", "100"),)),
                (9, (("2026-11-01", "100"),)),
                (10, (("2026-11-01", "100"),)),
                (11, (("2026-11-08", "101"),)),
            ):
                capture_current(path, fetcher=lambda _, items=values: _response(*items),
                                poll_time=_utc(poll_day), received_at=_utc(poll_day, second=1))
            coverage = source_coverage(path, asof=_utc(11, hour=7))
            self.assertEqual(coverage.missing_poll_days, 0)
            self.assertEqual(coverage.missing_or_late_observation_days, 2)
            self.assertFalse(coverage.five_year_continuous_source)


if __name__ == "__main__":
    unittest.main()
