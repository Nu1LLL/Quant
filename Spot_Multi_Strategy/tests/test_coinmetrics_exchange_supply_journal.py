"""Synthetic point-in-time observation and revision tests (no network)."""

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from research.crypto_blackbox.coinmetrics_exchange_supply_journal import (
    JournalError, capture_current, read_asof,
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


if __name__ == "__main__":
    unittest.main()
