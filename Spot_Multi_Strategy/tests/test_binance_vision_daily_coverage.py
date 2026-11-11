"""Synthetic read-only coverage checks; no price, strategy or market results."""

from datetime import date, datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from research.crypto_blackbox.binance_vision_daily_coverage import CoverageError, audit
from research.crypto_blackbox.binance_vision_daily_prospective import (
    ACTIVATION_DATE, SYMBOLS, _archive_urls,
)


def _write_day(root: Path, day: date, *, first_id: int = 1000,
               bad_hour: bool = False, captured_late: bool = False) -> None:
    sources = []
    for symbol in SYMBOLS:
        url, _, _ = _archive_urls(day, symbol)
        hours = []
        for hour in range(24):
            ident = first_id + hour + (1 if bad_hour and hour >= 12 else 0)
            hours.append({"hour": hour, "status": "observed", "count": 1,
                          "buy_count": 1, "sell_count": 0,
                          "buy_quote_micros": 1000000, "sell_quote_micros": 0,
                          "first_id": ident, "last_id": ident})
        sources.append({"symbol": symbol, "url": url, "zip_sha256": "a" * 64,
                        "zip_bytes": 500, "rows": 24, "hours": hours})
    captured = datetime.combine(day + timedelta(days=2 if captured_late else 1),
                                datetime.min.time(), tzinfo=timezone.utc)
    body = {"schema": 1, "source": "binance_vision_spot_daily_aggTrades",
            "day": day.isoformat(), "captured_at_utc": captured.isoformat(),
            "independent_executable_oos": False, "sources": sources}
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{day.isoformat()}.json.gz").write_bytes(gzip.compress(
        json.dumps(body, sort_keys=True).encode(), mtime=0))


class BinanceVisionCoverageTests(unittest.TestCase):
    def test_empty_prelaunch_and_one_complete_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(audit(root, as_of=ACTIVATION_DATE)["expected_days"], 0)
            _write_day(root, ACTIVATION_DATE)
            report = audit(root, as_of=ACTIVATION_DATE + timedelta(days=1))
            self.assertEqual(report["complete_days"], 1)
            self.assertEqual(report["longest_complete_run_days"], 1)
            self.assertFalse(report["five_year_continuous_source"])
            self.assertFalse(report["independent_executable_oos_proven"])

    def test_missing_day_breaks_run_without_erasing_later_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_day(root, ACTIVATION_DATE)
            later = ACTIVATION_DATE + timedelta(days=2)
            _write_day(root, later, first_id=2000)
            report = audit(root, as_of=later + timedelta(days=1))
            self.assertEqual(report["complete_days"], 2)
            self.assertEqual(report["longest_complete_run_days"], 1)
            self.assertIn((ACTIVATION_DATE + timedelta(days=1)).isoformat(),
                          report["first_bad_days"])

    def test_cross_day_id_gap_rejects_second_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_day(root, ACTIVATION_DATE)
            second = ACTIVATION_DATE + timedelta(days=1)
            _write_day(root, second, first_id=2000)
            report = audit(root, as_of=second + timedelta(days=1))
            self.assertEqual(report["complete_days"], 1)
            self.assertEqual(report["first_bad_days"], [second.isoformat()])

    def test_malformed_hour_or_future_capture_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_day(root, ACTIVATION_DATE, bad_hour=True)
            with self.assertRaises(CoverageError):
                audit(root, as_of=ACTIVATION_DATE + timedelta(days=1))
            _write_day(root, ACTIVATION_DATE, captured_late=True)
            with self.assertRaises(CoverageError):
                audit(root, as_of=ACTIVATION_DATE + timedelta(days=1))


if __name__ == "__main__":
    unittest.main()
