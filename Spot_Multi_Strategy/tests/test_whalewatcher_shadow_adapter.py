"""Synthetic contract tests; no account, market, or strategy-result data."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.crypto_blackbox.whalewatcher_shadow_adapter import (
    read_whalewatcher_shadow_snapshot,
)


NOW_MS = 1_800_000_000_000
GENERATED_MS = NOW_MS - 1_000
SOURCE_MS = NOW_MS - 60_000


def snapshot():
    return {
        "schema_version": 2,
        "generated_at_ms": GENERATED_MS,
        "signal_role": "external_intelligence_shadow",
        "execution_eligible": False,
        "operator_execution_eligible": False,
        "promotion_eligible": False,
        "manual_review_required": True,
        "performance_calibration_used": False,
        "source_health": {
            "schema_version": 1,
            "generated_at_ms": GENERATED_MS,
            "signal_role": "external_intelligence_shadow",
            "execution_eligible": False,
            "credentials_present_in_state": False,
            "status": "ready",
            "configured_source_count": 1,
            "ready_source_count": 1,
            "any_source_ready": True,
            "all_configured_sources_ready": True,
        },
        "readiness": {
            "status": "ready",
            "configured_source_count": 1,
            "ready_source_count": 1,
            "any_source_ready": True,
            "all_configured_sources_ready": True,
        },
        "symbols": {
            "BTC": {
                "manual_review_required": True,
                "execution_eligible": False,
                "observations": [{
                    "observation_id": "a" * 32,
                    "source": "vergex",
                    "fact_group_id": "chain:tx:" + "b" * 64,
                    "available_at_ms": SOURCE_MS,
                    "side": "short",
                    "verification_status": "verified",
                    "consensus_eligible": True,
                    "execution_eligible": False,
                    "source_url": "https://example.test/private?token=never-copy",
                    "summary": "PRIVATE SUMMARY SENTINEL",
                    "notional_usd": 10**9,
                    "study": {"post_event_return_pct": 80},
                    "review": {"trade_action": "SELL NOW"},
                }],
            },
            "ETH": {
                "manual_review_required": True,
                "execution_eligible": False,
                "observations": [],
            },
        },
    }


class WhaleWatcherShadowAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "shadow.json"
        self.clock = patch(
            "research.crypto_blackbox.whalewatcher_shadow_adapter.time.time_ns",
            return_value=NOW_MS * 1_000_000,
        )
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def write(self, payload):
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def read(self, symbol="BTC"):
        return read_whalewatcher_shadow_snapshot(self.path, symbol)

    def test_valid_snapshot_is_shadow_only_and_never_backfills_verification(self):
        self.write(snapshot())
        result = self.read("BTCUSDT")
        self.assertEqual(result.status, "fresh")
        self.assertEqual(result.symbol, "BTC")
        self.assertTrue(result.research_only)
        self.assertFalse(result.execution_eligible)
        self.assertFalse(result.promotion_eligible)
        self.assertEqual(len(result.claims), 1)
        claim = result.claims[0]
        self.assertEqual(claim.asserted_side, "short")
        self.assertFalse(claim.execution_eligible)
        self.assertEqual(claim.source_available_at_ms, SOURCE_MS)
        # WhaleWatcher v2 omits verified_at_ms.  The claim only becomes
        # available when Quant actually reads it, not at source publication.
        self.assertEqual(claim.quant_available_at_ms, NOW_MS)
        self.assertGreater(claim.quant_available_at_ms, result.snapshot_generated_at_ms)
        self.assertEqual(len(result.snapshot_sha256), 64)

        copied = json.dumps(asdict(result))
        for forbidden in (
            "never-copy", "PRIVATE SUMMARY SENTINEL", "post_event_return_pct",
            "SELL NOW", "notional_usd", "trade_action", "source_url",
        ):
            self.assertNotIn(forbidden, copied)
        self.assertNotIn("action", asdict(claim))

    def test_missing_empty_and_stale_snapshot_fail_closed(self):
        self.assertEqual(self.read().status, "unavailable")
        self.path.write_bytes(b"")
        self.assertEqual(self.read().claims, ())
        stale = snapshot()
        stale["generated_at_ms"] = NOW_MS - 600_001
        self.write(stale)
        self.assertEqual(self.read().status, "unavailable")
        future = snapshot()
        future["generated_at_ms"] = NOW_MS + 1
        self.write(future)
        self.assertEqual(self.read().status, "unavailable")

    def test_rejects_promoted_or_malformed_snapshot_as_a_whole(self):
        for change in (
            ("execution_eligible", True),
            ("operator_execution_eligible", True),
            ("promotion_eligible", True),
            ("manual_review_required", False),
            ("performance_calibration_used", True),
            ("schema_version", 1),
        ):
            with self.subTest(change=change):
                payload = snapshot()
                payload[change[0]] = change[1]
                self.write(payload)
                result = self.read()
                self.assertEqual((result.status, result.claims), ("unavailable", ()))

        for change in (
            ("available_at_ms", NOW_MS + 1),
            ("available_at_ms", "bad"),
            ("consensus_eligible", "yes"),
            ("execution_eligible", True),
            ("verification_status", "partially_verified"),
            ("fact_group_id", "shared_unknown"),
        ):
            with self.subTest(change=change):
                payload = snapshot()
                payload["symbols"]["BTC"]["observations"][0][change[0]] = change[1]
                self.write(payload)
                self.assertEqual(self.read().status, "unavailable")

    def test_unverified_claim_is_excluded_without_promoting_it(self):
        payload = snapshot()
        row = payload["symbols"]["BTC"]["observations"][0]
        row["verification_status"] = "pending"
        row["consensus_eligible"] = False
        self.write(payload)
        result = self.read()
        self.assertEqual(result.status, "fresh_no_claims")
        self.assertEqual(result.claims, ())

    def test_unconfigured_or_degraded_sources_are_unavailable_not_empty_news(self):
        for key, value in (
            ("configured_source_count", 0),
            ("ready_source_count", 0),
            ("status", "unconfigured"),
            ("status", "degraded"),
            ("any_source_ready", False),
            ("all_configured_sources_ready", False),
            ("generated_at_ms", GENERATED_MS - 600_001),
        ):
            with self.subTest(key=key, value=value):
                payload = snapshot()
                payload["symbols"]["BTC"]["observations"] = []
                payload["source_health"][key] = value
                if key in payload["readiness"]:
                    payload["readiness"][key] = value
                self.write(payload)
                self.assertEqual(self.read().status, "unavailable")
        payload = snapshot()
        del payload["source_health"]
        self.write(payload)
        self.assertEqual(self.read().status, "unavailable")

    def test_unsafe_peer_symbol_invalidates_the_whole_snapshot(self):
        payload = snapshot()
        payload["symbols"]["ETH"]["execution_eligible"] = True
        self.write(payload)
        self.assertEqual(self.read("BTC").status, "unavailable")

    def test_duplicate_keys_nonfinite_and_invalid_arguments(self):
        self.path.write_text('{"schema_version":2,"schema_version":2}', encoding="utf-8")
        self.assertEqual(self.read().status, "unavailable")
        self.path.write_text('{"schema_version":NaN}', encoding="utf-8")
        self.assertEqual(self.read().status, "unavailable")
        with self.assertRaises(ValueError):
            self.read("SOL")
        with self.assertRaises(ValueError):
            read_whalewatcher_shadow_snapshot(self.path, "BTC", max_age_ms=600_001)


if __name__ == "__main__":
    unittest.main()
