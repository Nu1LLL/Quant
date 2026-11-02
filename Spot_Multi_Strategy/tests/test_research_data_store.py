import sys
import unittest
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))

from research.data_store import MarketDataStore, utc_us
from research.registry import ResearchRegistry


class MarketDataStoreTests(unittest.TestCase):
    def setUp(self):
        self.registry = ResearchRegistry(":memory:")
        self.store = MarketDataStore(self.registry)
        self.quote = {
            "provider": "sample", "dataset_version": "1", "venue": "X",
            "instrument": "TEST", "event_id": "1", "market": "equity",
            "event_time": "2026-01-01T00:00:00Z",
            "available_time": "2026-01-01T00:00:01Z",
            "bid": 100.0, "ask": 101.0, "bid_size": 10, "ask_size": 20,
        }

    def tearDown(self):
        self.registry.close()

    def decision(self, seconds):
        return utc_us("2026-01-01T00:00:00Z") + seconds * 1_000_000

    def test_quote_availability_staleness_and_capability(self):
        self.assertEqual(self.store.ingest_quote(self.quote), "accepted")
        self.assertIsNone(
            self.store.latest_quote("sample", "1", "X", "TEST", self.decision(0), 10_000_000)
        )
        self.assertIsNotNone(
            self.store.latest_quote("sample", "1", "X", "TEST", self.decision(1), 10_000_000)
        )
        self.assertIsNone(
            self.store.latest_quote(
                "sample", "1", "X", "TEST", self.decision(2), 10_000_000,
                required_capabilities=("l2_depth",),
            )
        )

    def test_conflicting_duplicate_is_quarantined(self):
        self.store.ingest_quote(self.quote)
        self.assertEqual(self.store.ingest_quote(self.quote), "duplicate")
        changed = dict(self.quote, bid=99.0)
        self.assertEqual(self.store.ingest_quote(changed), "quarantined")

    def test_bars_are_point_in_time(self):
        bar = {
            "provider": "sample", "dataset_version": "1", "venue": "X",
            "instrument": "TEST", "event_id": "bar-1", "market": "crypto",
            "event_time": "2026-01-01T00:00:00Z",
            "available_time": "2026-01-01T04:00:01Z", "interval": "4h",
            "open": 100.0, "high": 103.0, "low": 99.0, "close": 102.0,
            "volume": 1000.0,
        }
        self.assertEqual(self.store.ingest_bar(bar), "accepted")
        early = utc_us("2026-01-01T04:00:00Z")
        late = utc_us("2026-01-01T04:00:01Z")
        self.assertTrue(self.store.bars_asof("sample", "1", "X", "TEST", early).empty)
        self.assertEqual(len(self.store.bars_asof("sample", "1", "X", "TEST", late)), 1)

        revision = dict(
            bar, event_id="bar-1-revision", close=101.0,
            available_time="2026-01-01T05:00:00Z"
        )
        self.assertEqual(self.store.ingest_bar(revision), "accepted")
        before_revision = self.store.bars_asof(
            "sample", "1", "X", "TEST", utc_us("2026-01-01T04:30:00Z")
        )
        after_revision = self.store.bars_asof(
            "sample", "1", "X", "TEST", utc_us("2026-01-01T05:00:00Z")
        )
        self.assertEqual(float(before_revision.iloc[0]["close"]), 102.0)
        self.assertEqual(float(after_revision.iloc[0]["close"]), 101.0)


if __name__ == "__main__":
    unittest.main()
