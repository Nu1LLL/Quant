"""Synthetic, non-trading tests for fixed multi-coin shadow diversification."""

from datetime import datetime, timedelta, timezone
import unittest

from research.crypto_blackbox.archetypes import Candidate
from research.crypto_blackbox.matching import MatchDecision
from research.crypto_blackbox.shadow_portfolio import (
    AllocationConfig, allocate_shadow_portfolio,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def ranked(symbol, score, *, side="LONG", archetype="flow", weight=0.2):
    candidate = Candidate(
        archetype, "crypto_perpetual", "SYNTH", symbol, side,
        weight if side == "LONG" else -weight, 60, "synthetic",
        decision_at=NOW,
    )
    return MatchDecision(
        candidate, "SHADOW_RANKED", ("synthetic_report_only",),
        evidence_id=f"evidence-{symbol}-{archetype}-{side}",
        net_lower_bound_bps=score,
    )


class ShadowPortfolioTests(unittest.TestCase):
    def test_predeclared_rank_and_cluster_caps_keep_cash(self):
        decisions = [ranked("SOLUSDT", 4), ranked("BTCUSDT", 10),
                     ranked("ETHUSDT", 8), ranked("XRPUSDT", 2)]
        result = allocate_shadow_portfolio(
            decisions, NOW,
            cluster_by_symbol={"BTCUSDT": "BTC", "ETHUSDT": "ETH"},
        )
        self.assertEqual([a.symbol for a in result.allocations],
                         ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        self.assertAlmostEqual(result.gross_weight, 0.6)
        self.assertAlmostEqual(result.cash_weight, 0.4)
        self.assertAlmostEqual(result.allocations[2].target_weight, 0.2)
        self.assertIn("XRPUSDT:diversification_cap", result.blocked)
        self.assertTrue(result.research_only)
        self.assertFalse(result.deployable)
        self.assertFalse(hasattr(result, "orders"))

    def test_unknown_coins_share_alt_cluster_conservatively(self):
        result = allocate_shadow_portfolio(
            [ranked("SOLUSDT", 8), ranked("XRPUSDT", 6), ranked("DOGEUSDT", 4)], NOW,
        )
        self.assertAlmostEqual(result.gross_weight, 0.3)
        self.assertEqual(len(result.allocations), 2)
        self.assertAlmostEqual(result.allocations[0].target_weight, 0.2)
        self.assertAlmostEqual(result.allocations[1].target_weight, 0.1)
        self.assertIn("DOGEUSDT:diversification_cap", result.blocked)

    def test_opposing_directions_block_and_duplicate_exposure_is_not_added(self):
        conflict = allocate_shadow_portfolio(
            [ranked("BTCUSDT", 9), ranked("BTCUSDT", 8, side="SHORT")], NOW,
        )
        self.assertEqual(conflict.allocations, ())
        self.assertIn("BTCUSDT:opposing_directions", conflict.blocked)
        duplicate = allocate_shadow_portfolio(
            [ranked("BTCUSDT", 9), ranked("BTCUSDT", 7, archetype="rising_tide")], NOW,
        )
        self.assertEqual(len(duplicate.allocations), 1)
        self.assertIn("BTCUSDT:duplicate_exposure", duplicate.blocked)

    def test_failed_or_missing_evidence_never_allocates(self):
        failed = MatchDecision(
            ranked("BTCUSDT", 9).candidate, "GATE_FAILED", ("sharpe",),
            evidence_id="failed", net_lower_bound_bps=100,
        )
        result = allocate_shadow_portfolio([failed], NOW)
        self.assertEqual(result.allocations, ())
        self.assertEqual(result.cash_weight, 1.0)
        self.assertIn("BTCUSDT:GATE_FAILED", result.blocked)

    def test_reject_stale_candidate_and_invalid_config(self):
        with self.assertRaisesRegex(ValueError, "same-time"):
            allocate_shadow_portfolio([ranked("BTCUSDT", 9)], NOW + timedelta(seconds=1))
        with self.assertRaisesRegex(ValueError, "finite"):
            AllocationConfig(max_portfolio_gross=float("nan"))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            allocate_shadow_portfolio([], NOW, cluster_by_symbol={"BTCUSDT": ""})


if __name__ == "__main__":
    unittest.main()
