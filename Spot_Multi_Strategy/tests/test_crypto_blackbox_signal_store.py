"""Synthetic-only SQLite observation-store tests."""

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.crypto_blackbox import CryptoObservation, classify_observation
from research.crypto_blackbox.signal_store import SignalStore


BASE = datetime(2026, 3, 1, 0, 5, tzinfo=timezone.utc)


def observation(*, event_id="event-1", event_time=BASE, source="synthetic",
                symbol="BTCUSDT", market="crypto_perpetual"):
    return CryptoObservation(
        event_id=event_id, source=source, source_version="v1",
        license_id="test-only", market=market, venue="BINANCE", symbol=symbol,
        family="flow_volume_position", window_start=event_time - timedelta(minutes=5),
        event_time=event_time, available_time=event_time + timedelta(seconds=2),
        expires_at=event_time + timedelta(minutes=5), horizon_seconds=300,
        observed_fields={
            "aggressor_buy_share_5m": 0.6,
            "quote_volume_5m_usd": 300.0,
            "quote_volume_baseline_5m_usd": 100.0,
            "trade_count_5m": 2,
        },
    )


class SignalStoreTests(unittest.TestCase):
    def test_asof_requires_both_observation_and_classification_visibility(self):
        item = observation()
        label = classify_observation(item, BASE + timedelta(minutes=1))
        with SignalStore(":memory:") as store:
            store.append_observation(item)
            self.assertEqual(store.list_asof(BASE + timedelta(seconds=3)), ())
            store.append_classification(label)
            self.assertEqual(store.list_asof(BASE + timedelta(seconds=1)), ())
            self.assertEqual(store.list_asof(BASE + timedelta(seconds=3)), ())
            visible = store.list_asof(BASE + timedelta(minutes=1))
            self.assertEqual(len(visible), 1)
            self.assertEqual(visible[0].observation.event_id, item.event_id)
            self.assertEqual(visible[0].observation.source_version, "v1")
            self.assertEqual(visible[0].observation.license_id, "test-only")
            self.assertEqual(visible[0].classification.label, "FIRST_ALERT")
            self.assertFalse(visible[0].classification.tradable)
            self.assertNotIn("pnl", str(visible[0].observation.observed_fields).lower())

    def test_duplicate_and_ambiguous_source_identity_fail_closed(self):
        item = observation()
        with SignalStore() as store:
            store.append_observation(item)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                store.append_observation(item)
            other = observation(source="another-provider")
            store.append_observation(other)
            label = classify_observation(item, item.available_time)
            with self.assertRaisesRegex(ValueError, "unambiguous"):
                store.append_classification(label)

    def test_one_label_only_and_expiry_consistency(self):
        item = observation()
        with SignalStore() as store:
            store.append_observation(item)
            early = classify_observation(item, item.available_time)
            late = classify_observation(item, item.expires_at)
            with self.assertRaisesRegex(ValueError, "expiry"):
                store.append_classification(late.__class__(
                    event_id=late.event_id, market=late.market, venue=late.venue,
                    symbol=late.symbol, family=late.family, label="FIRST_ALERT",
                    reasons=late.reasons, asof_time=late.asof_time,
                ))
            store.append_classification(early)
            with self.assertRaisesRegex(ValueError, "already exists"):
                store.append_classification(early)

    def test_sqlite_triggers_reject_updates_and_deletes(self):
        item = observation()
        with SignalStore() as store:
            store.append_observation(item)
            store.append_classification(classify_observation(item, item.available_time))
            for query in (
                "UPDATE observations SET license_id='other' WHERE id=1",
                "DELETE FROM observations WHERE id=1",
                "UPDATE classifications SET label='NO_ALERT' WHERE id=1",
                "DELETE FROM classifications WHERE id=1",
            ):
                with self.subTest(query=query), self.assertRaises(sqlite3.IntegrityError):
                    store.db.execute(query)
                store.db.rollback()

    def test_file_store_reopen_filter_limit_and_no_raw_assets(self):
        first = observation()
        second = observation(event_id="event-2", event_time=BASE + timedelta(minutes=5),
                             symbol="ETHUSDT", market="crypto_spot")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.sqlite3"
            with SignalStore(path) as store:
                for item in (first, second):
                    store.append_observation(item)
                    store.append_classification(classify_observation(item, item.available_time))
            with SignalStore(path) as store:
                at = BASE + timedelta(minutes=6)
                self.assertEqual(len(store.list_asof(at)), 2)
                self.assertEqual(len(store.list_asof(at, market="crypto_spot")), 1)
                self.assertEqual(len(store.list_asof(at, limit=1)), 1)
                self.assertEqual(store.list_asof(at, limit=1)[0].observation.symbol, "ETHUSDT")
                with self.assertRaises(ValueError):
                    store.list_asof(at, market="equity")
                with self.assertRaises(ValueError):
                    store.list_asof(at, limit=1001)
                columns = {
                    row[1] for row in store.db.execute("PRAGMA table_info(observations)")
                }
                self.assertNotIn("raw_packet", columns)
                self.assertNotIn("image", columns)
                self.assertNotIn("return", columns)
            self.assertTrue(path.exists())
        self.assertFalse(path.exists())

    def test_machine_code_reasons_only_and_no_future_label(self):
        item = observation()
        with SignalStore() as store:
            store.append_observation(item)
            early = classify_observation(item, item.available_time)
            invalid = early.__class__(
                event_id=early.event_id, market=early.market, venue=early.venue,
                symbol=early.symbol, family=early.family, label=early.label,
                reasons=("健康截图 data:image/png;base64",),
                asof_time=early.asof_time,
            )
            with self.assertRaisesRegex(ValueError, "machine codes"):
                store.append_classification(invalid)
            outcome_reason = early.__class__(
                event_id=early.event_id, market=early.market, venue=early.venue,
                symbol=early.symbol, family=early.family, label=early.label,
                reasons=("future_return_positive",), asof_time=early.asof_time,
            )
            with self.assertRaisesRegex(ValueError, "outcome-derived"):
                store.append_classification(outcome_reason)
            before_availability = early.__class__(
                event_id=early.event_id, market=early.market, venue=early.venue,
                symbol=early.symbol, family=early.family, label=early.label,
                reasons=early.reasons, asof_time=item.event_time,
            )
            with self.assertRaisesRegex(ValueError, "precedes observation"):
                store.append_classification(before_availability)


if __name__ == "__main__":
    unittest.main()
