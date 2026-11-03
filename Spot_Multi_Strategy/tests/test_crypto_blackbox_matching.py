"""Synthetic-only tests for exact, causal, non-executable evidence matching."""

from datetime import datetime, timedelta, timezone
import unittest

from research.crypto_blackbox.archetypes import Candidate
from research.crypto_blackbox.matching import (
    EvidenceRecord, rank_shadow_candidates, reported_gate_failures,
)
from research.crypto_blackbox.schemas import ObservationClassification


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candidate(symbol="BTCUSDT", venue="SYNTH", side="LONG", horizon=60):
    return Candidate("flow", "crypto_perpetual", venue, symbol, side,
                     0.2 if side == "LONG" else -0.2, horizon, "synthetic",
                     decision_at=NOW)


def evidence(**changes):
    fields = dict(
        evidence_id="sealed-result-1", prereg_commit="abcdef0",
        result_commit="abcdef1", license_id="synthetic-only",
        market="crypto_perpetual", venue="SYNTH", symbol="BTCUSDT",
        archetype="flow", side="LONG", horizon_minutes=60,
        preregistered_at=datetime(2019, 12, 31, tzinfo=timezone.utc),
        history_start=datetime(2018, 1, 1, tzinfo=timezone.utc),
        history_end=datetime(2025, 10, 1, tzinfo=timezone.utc),
        oos_start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        oos_end=datetime(2025, 10, 1, tzinfo=timezone.utc),
        available_at=datetime(2025, 10, 2, tzinfo=timezone.utc),
        expires_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
        mean_gross_bps=20.0, roundtrip_cost_bps=5.0,
        stderr_bps=2.0, trades=100, cagr=0.45, sharpe=1.7,
        max_drawdown=0.18, daily_profit_factor=1.4, leverage=1.0,
        walk_forward_years=5, walk_forward_success_fraction=0.80,
        monte_carlo_paths=5_000, monte_carlo_block_days=21,
        monte_carlo_joint_pass_fraction=0.06,
        independent_oos=True, annual_walk_forward=True, walk_forward_passed=True,
        regimes_passed=True, block_monte_carlo_passed=True,
        data_integrity_passed=True, fees_slippage_included=True,
        funding_included_if_applicable=True, executable_for_10000_usd=True,
    )
    fields.update(changes)
    return EvidenceRecord(**fields)


class EvidenceMatchingTests(unittest.TestCase):
    def test_reported_gate_pass_only_ranks_shadow_not_order(self):
        label = ObservationClassification(
            "event-1", "crypto_perpetual", "SYNTH", "BTCUSDT",
            "flow_volume_position", "FIRST_ALERT", ("synthetic",), NOW,
        )
        rows = rank_shadow_candidates([candidate()], [evidence()], NOW, [label])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.status, "SHADOW_RANKED")
        self.assertAlmostEqual(row.net_lower_bound_bps, 11.08)
        self.assertEqual(row.alert_label, "FIRST_ALERT")
        self.assertTrue(row.research_only)
        self.assertFalse(row.deployable)
        self.assertFalse(hasattr(row, "order"))
        self.assertFalse(hasattr(row, "quantity"))

    def test_exact_market_venue_symbol_side_horizon_required(self):
        for changed in (candidate(symbol="ETHUSDT"), candidate(venue="OTHER"),
                        candidate(side="SHORT"), candidate(horizon=30)):
            with self.subTest(candidate=changed):
                result = rank_shadow_candidates([changed], [evidence()], NOW)[0]
                self.assertEqual(result.status, "NO_EVIDENCE")
        future = evidence(available_at=NOW + timedelta(days=1),
                          expires_at=NOW + timedelta(days=2))
        self.assertEqual(rank_shadow_candidates([candidate()], [future], NOW)[0].status,
                         "NO_EVIDENCE")

    def test_fail_closed_on_multiple_results_and_failed_gates(self):
        records = [evidence(), evidence(evidence_id="sealed-result-2",
                                         result_commit="abcdef2", mean_gross_bps=100.0)]
        ambiguous = rank_shadow_candidates([candidate()], records, NOW)[0]
        self.assertEqual(ambiguous.status, "AMBIGUOUS_EVIDENCE")
        bad = evidence(sharpe=1.4, max_drawdown=0.25,
                       independent_oos=False, regimes_passed=False)
        self.assertEqual(rank_shadow_candidates([candidate()], [bad], NOW)[0].status,
                         "GATE_FAILED")
        failures = reported_gate_failures(bad)
        self.assertIn("sharpe_not_above_1_5", failures)
        self.assertIn("drawdown_above_20_percent", failures)
        self.assertIn("independent_oos_not_passed", failures)
        self.assertIn("regimes_passed_not_passed", failures)

    def test_net_lower_bound_accounts_for_cost_and_uncertainty(self):
        poor = evidence(mean_gross_bps=10.0, roundtrip_cost_bps=8.0,
                        stderr_bps=2.0)
        row = rank_shadow_candidates([candidate()], [poor], NOW)[0]
        self.assertEqual(row.status, "NO_POSITIVE_NET_EDGE")
        self.assertLess(row.net_lower_bound_bps, 0)

    def test_reject_future_classification_and_noncausal_evidence(self):
        future = ObservationClassification(
            "event-2", "crypto_perpetual", "SYNTH", "BTCUSDT",
            "flow_volume_position", "FIRST_ALERT", ("synthetic",),
            NOW + timedelta(seconds=1),
        )
        with self.assertRaisesRegex(ValueError, "future classification"):
            rank_shadow_candidates([candidate()], [evidence()], NOW, [future])
        with self.assertRaisesRegex(ValueError, "exact decision time"):
            rank_shadow_candidates([candidate()], [evidence()], NOW + timedelta(seconds=1))
        with self.assertRaisesRegex(ValueError, "not causal"):
            evidence(oos_end=NOW + timedelta(days=1))
        with self.assertRaisesRegex(ValueError, "must differ"):
            evidence(result_commit="abcdef0")
        with self.assertRaisesRegex(ValueError, "must be crypto"):
            evidence(market="equity")

    def test_five_year_history_and_oos_are_hard_gates(self):
        short = evidence(history_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
                         oos_start=datetime(2025, 1, 1, tzinfo=timezone.utc))
        self.assertIn("history_under_five_years", reported_gate_failures(short))
        short_oos = evidence(oos_start=datetime(2025, 6, 1, tzinfo=timezone.utc))
        self.assertIn("oos_under_five_years", reported_gate_failures(short_oos))
        self.assertEqual(rank_shadow_candidates([candidate()], [short], NOW)[0].status,
                         "GATE_FAILED")
        retrospectively_registered = evidence(
            preregistered_at=datetime(2025, 9, 1, tzinfo=timezone.utc)
        )
        self.assertIn("historical_holdout_not_prospective_oos",
                      reported_gate_failures(retrospectively_registered))

    def test_protocol_gate_rejects_leveraged_cost_omissions_and_weak_mc(self):
        bad = evidence(leverage=3.0, walk_forward_years=4,
                       monte_carlo_paths=100, monte_carlo_joint_pass_fraction=0.0,
                       fees_slippage_included=False,
                       funding_included_if_applicable=False,
                       executable_for_10000_usd=False)
        failures = reported_gate_failures(bad)
        self.assertIn("base_signal_not_tested_at_one_x", failures)
        self.assertIn("annual_walk_forward_folds_not_sufficient", failures)
        self.assertIn("block_monte_carlo_protocol_not_passed", failures)
        self.assertIn("fees_slippage_included_not_passed", failures)
        self.assertIn("funding_included_if_applicable_not_passed", failures)
        self.assertIn("executable_for_10000_usd_not_passed", failures)

    def test_experiment_specific_success_rules_are_not_hardcoded(self):
        # Success fractions are reported for audit, but each experiment must
        # freeze its own success thresholds in its preregistration.
        custom = evidence(walk_forward_success_fraction=0.60,
                          monte_carlo_joint_pass_fraction=0.03)
        self.assertEqual(reported_gate_failures(custom), ())
        self.assertEqual(rank_shadow_candidates([candidate()], [custom], NOW)[0].status,
                         "SHADOW_RANKED")
        failed = evidence(walk_forward_passed=False,
                          block_monte_carlo_passed=False,
                          walk_forward_success_fraction=0.0,
                          monte_carlo_joint_pass_fraction=0.0)
        self.assertIn("walk_forward_passed_not_passed",
                      reported_gate_failures(failed))
        self.assertIn("block_monte_carlo_passed_not_passed",
                      reported_gate_failures(failed))
        self.assertEqual(rank_shadow_candidates([candidate()], [failed], NOW)[0].status,
                         "GATE_FAILED")

    def test_sort_only_proven_candidates_and_never_select_on_missing_results(self):
        eth = candidate(symbol="ETHUSDT")
        eth_evidence = evidence(symbol="ETHUSDT", evidence_id="eth",
                                mean_gross_bps=16.0)
        rows = rank_shadow_candidates([eth, candidate(), candidate("SOLUSDT")],
                                      [eth_evidence, evidence()], NOW)
        self.assertEqual([row.status for row in rows],
                         ["SHADOW_RANKED", "SHADOW_RANKED", "NO_EVIDENCE"])
        self.assertEqual(rows[0].candidate.symbol, "BTCUSDT")
        self.assertEqual(rows[-1].candidate.symbol, "SOLUSDT")


if __name__ == "__main__":
    unittest.main()
