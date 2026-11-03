"""Synthetic-only tests for crypto observation intake; no market/PnL data."""

import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from research.crypto_blackbox import CryptoObservation, SignalIntake, classify_observation


BASE = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)


def observation(*, event_id="test-1", event_time=BASE, market="crypto_perpetual",
                venue="BINANCE", symbol="BTCUSDT", family="flow_volume_position",
                fields=None, **overrides):
    data = {
        "event_id": event_id,
        "source": "synthetic",
        "source_version": "v1",
        "license_id": "test-only",
        "market": market,
        "venue": venue,
        "symbol": symbol,
        "family": family,
        "window_start": event_time - timedelta(minutes=5),
        "event_time": event_time,
        "available_time": event_time + timedelta(seconds=2),
        "expires_at": event_time + timedelta(minutes=5),
        "horizon_seconds": 60,
        "observed_fields": fields if fields is not None else {
            "aggressor_buy_share_5m": 0.60,
            "quote_volume_5m_usd": 300.0,
            "quote_volume_baseline_5m_usd": 100.0,
            "open_interest_change_pct_5m": -0.46,
        },
    }
    data.update(overrides)
    return CryptoObservation(**data)


class CryptoSignalIntakeTests(unittest.TestCase):
    def test_crypto_only_provenance_and_temporal_contract(self):
        item = observation()
        self.assertEqual(item.market, "crypto_perpetual")
        self.assertEqual(item.license_id, "test-only")
        with self.assertRaises(FrozenInstanceError):
            item.market = "equity"
        with self.assertRaises(TypeError):
            item.observed_fields["aggressor_buy_share_5m"] = 0.9
        for change in ({"market": "equity"}, {"license_id": ""},
                       {"event_id": "bad id"}, {"symbol": "btc-usdt"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                observation(**change)

    def test_reject_unavailable_future_stale_and_non_utc(self):
        item = observation()
        with self.assertRaisesRegex(ValueError, "not yet visible"):
            classify_observation(item, BASE)
        stale = classify_observation(item, BASE + timedelta(minutes=5))
        self.assertEqual(stale.label, "STALE")
        self.assertFalse(stale.tradable)
        for change in (
            {"event_time": BASE.replace(tzinfo=None)},
            {"available_time": BASE - timedelta(seconds=1)},
            {"expires_at": BASE + timedelta(seconds=1)},
            {"window_start": BASE - timedelta(minutes=4)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                observation(**change)

    def test_unknown_and_promotional_outcomes_are_forbidden(self):
        bad_fields = (
            {"post_alert_gain_pct": 54.24},
            {"future_return_5m": 0.1},
            {"realized_pnl": 10},
            {"prediction": 0.95},
        )
        for fields in bad_fields:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                observation(fields=fields)
        with self.assertRaises(ValueError):
            observation(fields={"aggressor_buy_share_5m": 1.01})
        with self.assertRaises(ValueError):
            observation(fields={"quote_volume_baseline_5m_usd": 0})
        with self.assertRaises(ValueError):
            observation(fields={"open_interest_change_pct_5m": float("nan")})
        with self.assertRaises(ValueError):
            observation(market="crypto_spot")
        spot = observation(market="crypto_spot", fields={
            "aggressor_buy_share_5m": 0.6,
            "quote_volume_5m_usd": 300.0,
            "quote_volume_baseline_5m_usd": 100.0,
        })
        self.assertEqual(spot.market, "crypto_spot")

    def test_first_then_continuation_then_first_after_one_hour(self):
        intake = SignalIntake()
        first = observation()
        first_label = intake.classify(first, first.available_time)
        self.assertEqual(first_label.label, "FIRST_ALERT")
        self.assertIn("open_interest_decreasing", first_label.reasons)
        second = observation(event_id="test-2", event_time=BASE + timedelta(minutes=5))
        second_label = intake.classify(second, second.available_time)
        self.assertEqual(second_label.label, "CONTINUATION")
        later = observation(event_id="test-3", event_time=BASE + timedelta(hours=1, minutes=5))
        later_label = intake.classify(later, later.available_time)
        self.assertEqual(later_label.label, "FIRST_ALERT")
        for label in (first_label, second_label, later_label):
            self.assertFalse(label.tradable)
            self.assertFalse(hasattr(label, "action"))
            self.assertFalse(hasattr(label, "expected_return"))

    def test_duplicate_and_out_of_order_rejected(self):
        intake = SignalIntake()
        first = observation()
        intake.classify(first, first.available_time)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            intake.classify(first, first.available_time)
        duplicate_window = observation(event_id="new-id")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            intake.classify(duplicate_window, duplicate_window.available_time)
        old = observation(event_id="old", event_time=BASE - timedelta(minutes=5))
        with self.assertRaisesRegex(ValueError, "event-time order"):
            intake.classify(old, old.available_time)

    def test_prior_alert_must_be_visible_and_decision_clock_monotone(self):
        first = observation()
        second = observation(event_id="second", event_time=BASE + timedelta(minutes=5))
        with self.assertRaisesRegex(ValueError, "supplied together"):
            classify_observation(second, second.available_time, BASE)
        with self.assertRaisesRegex(ValueError, "not visible"):
            classify_observation(
                second, second.available_time, BASE,
                second.available_time + timedelta(seconds=1),
            )
        intake = SignalIntake()
        intake.classify(first, BASE + timedelta(minutes=10))
        with self.assertRaisesRegex(ValueError, "cannot go backwards"):
            intake.classify(second, second.available_time)

    def test_different_sources_do_not_create_false_continuation(self):
        intake = SignalIntake()
        first = observation()
        self.assertEqual(intake.classify(first, first.available_time).label, "FIRST_ALERT")
        independent = observation(
            event_id="other-source", source="another-provider",
            event_time=BASE + timedelta(minutes=5),
        )
        self.assertEqual(
            intake.classify(independent, independent.available_time).label,
            "FIRST_ALERT",
        )

    def test_low_flow_and_missing_fields_are_non_alerts(self):
        low = observation(fields={
            "aggressor_buy_share_5m": 0.51,
            "quote_volume_5m_usd": 120.0,
            "quote_volume_baseline_5m_usd": 100.0,
        })
        label = classify_observation(low, low.available_time)
        self.assertEqual(label.label, "NO_ALERT")
        self.assertIn("buy_share_below_0_55", label.reasons)
        self.assertIn("volume_multiple_below_2", label.reasons)
        missing = observation(fields={"aggressor_buy_share_5m": 0.60})
        missing_label = classify_observation(missing, missing.available_time)
        self.assertEqual(missing_label.label, "NO_ALERT")
        self.assertTrue(any(reason.startswith("missing_") for reason in missing_label.reasons))
        overflow = observation(fields={
            "aggressor_buy_share_5m": 0.60,
            "quote_volume_5m_usd": 1e308,
            "quote_volume_baseline_5m_usd": 1e-308,
        })
        self.assertEqual(
            classify_observation(overflow, overflow.available_time).reasons,
            ("volume_multiple_nonfinite",),
        )

    def test_non_flow_family_is_recorded_but_not_flash_classified(self):
        item = observation(family="onchain", fields={})
        label = classify_observation(item, item.available_time)
        self.assertEqual(label.label, "NO_ALERT")
        self.assertEqual(label.reasons, ("family_not_supported_by_flow_classifier",))


if __name__ == "__main__":
    unittest.main()
