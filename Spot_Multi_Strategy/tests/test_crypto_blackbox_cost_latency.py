"""Synthetic-only cost and causal latency tests; no market data or orders."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest

from research.crypto_blackbox.cost_latency import (
    ACCOUNT_EQUITY_USD,
    CausalTimeline,
    CostInputs,
    assess_cost_latency,
)


START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def timeline() -> CausalTimeline:
    return CausalTimeline(
        signal_event_at=START,
        signal_available_at=START + timedelta(seconds=1),
        decision_at=START + timedelta(seconds=2),
        entry_submitted_at=START + timedelta(seconds=3),
        entry_filled_at=START + timedelta(seconds=5),
        exit_decision_at=START + timedelta(days=1, seconds=1),
        exit_submitted_at=START + timedelta(days=1, seconds=2),
        exit_filled_at=START + timedelta(days=1, seconds=5),
    )


def costs() -> CostInputs:
    return CostInputs(
        entry_fee_bps=Decimal("5"),
        exit_fee_bps=Decimal("5"),
        entry_spread_bps=Decimal("2"),
        exit_spread_bps=Decimal("2"),
        entry_slippage_bps=Decimal("3"),
        exit_slippage_bps=Decimal("3"),
        entry_impact_bps=Decimal("1"),
        exit_impact_bps=Decimal("1"),
        funding_adverse_bps=Decimal("2"),
        borrow_apr_bps=Decimal("365.25"),
        entry_delay_adverse_bps_per_second=Decimal("0.5"),
        exit_delay_adverse_bps_per_second=Decimal("0.25"),
    )


def assess(**overrides):
    fields = dict(
        timeline=timeline(), costs=costs(),
        position_notional_usd=Decimal("2500"),
        gross_edge_point_bps=Decimal("50"),
        gross_edge_uncertainty_haircut_bps=Decimal("10"),
        required_net_edge_bps=Decimal("2"),
    )
    fields.update(overrides)
    return assess_cost_latency(**fields)


class CostLatencyTests(unittest.TestCase):
    def test_round_trip_decomposition_and_ten_thousand_dollar_account(self):
        result = assess()
        self.assertEqual(ACCOUNT_EQUITY_USD, Decimal("10000"))
        self.assertEqual(result.gross_leverage, Decimal("0.25"))
        self.assertEqual(result.cost.entry_half_spread_bps, Decimal("1"))
        self.assertEqual(result.cost.exit_half_spread_bps, Decimal("1"))
        self.assertEqual(result.cost.borrow_bps, Decimal("1"))
        self.assertEqual(result.cost.entry_delay_adverse_bps, Decimal("2.5"))
        self.assertEqual(result.cost.exit_delay_adverse_bps, Decimal("1"))
        self.assertEqual(result.cost.total_bps, Decimal("26.5"))
        self.assertEqual(result.gross_edge_lower_bps, Decimal("40"))
        self.assertEqual(result.net_edge_lower_bps, Decimal("13.5"))
        self.assertEqual(result.net_edge_lower_usd, Decimal("3.375"))
        self.assertEqual(result.net_edge_lower_pct_account, Decimal("0.03375"))
        self.assertEqual(result.minimum_viable_gross_edge_bps, Decimal("38.5"))
        self.assertEqual(result.minimum_viable_gross_profit_usd, Decimal("9.625"))
        self.assertTrue(result.research_only)
        self.assertFalse(result.deployable)
        self.assertFalse(hasattr(result, "order"))

    def test_higher_cost_or_latency_worsens_lower_bound(self):
        original = assess()
        dearer = assess(costs=replace(costs(), entry_impact_bps=Decimal("6")))
        slower = assess(timeline=replace(
            timeline(), entry_submitted_at=START + timedelta(seconds=8),
            entry_filled_at=START + timedelta(seconds=10),
            exit_decision_at=START + timedelta(days=1, seconds=6),
            exit_submitted_at=START + timedelta(days=1, seconds=7),
            exit_filled_at=START + timedelta(days=1, seconds=10),
        ))
        self.assertEqual(dearer.net_edge_lower_bps,
                         original.net_edge_lower_bps - Decimal("5"))
        self.assertEqual(slower.net_edge_lower_bps,
                         original.net_edge_lower_bps - Decimal("2.5"))
        self.assertGreater(dearer.minimum_viable_gross_edge_bps,
                           original.minimum_viable_gross_edge_bps)

    def test_weak_gross_edge_yields_negative_net_lower_bound(self):
        result = assess(gross_edge_point_bps="4", gross_edge_uncertainty_haircut_bps="6")
        self.assertEqual(result.gross_edge_lower_bps, Decimal("-2"))
        self.assertEqual(result.net_edge_lower_bps, Decimal("-28.5"))
        self.assertLess(result.net_edge_lower_usd, Decimal("0"))

    def test_event_visibility_and_fill_chronology_fail_closed(self):
        base = timeline()
        for name, changes in (
            ("availability", {"signal_available_at": START - timedelta(microseconds=1)}),
            ("decision", {"decision_at": START + timedelta(seconds=6)}),
            ("submit", {"entry_submitted_at": START + timedelta(seconds=6)}),
            ("exit", {"exit_filled_at": START + timedelta(seconds=1)}),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "causal"):
                replace(base, **changes)
        with self.assertRaisesRegex(ValueError, "holding period"):
            CausalTimeline(
                signal_event_at=START, signal_available_at=START,
                decision_at=START, entry_submitted_at=START,
                entry_filled_at=START, exit_decision_at=START,
                exit_submitted_at=START, exit_filled_at=START,
            )
        with self.assertRaisesRegex(ValueError, "UTC"):
            replace(base, signal_event_at=START.replace(tzinfo=None))
        with self.assertRaisesRegex(ValueError, "UTC"):
            replace(base, signal_available_at=START.astimezone(
                timezone(timedelta(hours=1)),
            ))

    def test_negative_nonfinite_or_missing_costs_fail_closed(self):
        for name, bad in (
            ("entry_fee_bps", "-1"),
            ("funding_adverse_bps", "NaN"),
            ("borrow_apr_bps", "Infinity"),
            ("exit_slippage_bps", None),
            ("entry_impact_bps", 0.1),
        ):
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                replace(costs(), **{name: bad})
        with self.assertRaises(TypeError):
            CostInputs(entry_fee_bps="1")
        for name, bad in (
            ("position_notional_usd", "NaN"),
            ("gross_edge_point_bps", "-1"),
            ("gross_edge_uncertainty_haircut_bps", None),
            ("required_net_edge_bps", "-0.1"),
        ):
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                assess(**{name: bad})

    def test_no_position_or_above_five_times_account_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            assess(position_notional_usd="0")
        with self.assertRaisesRegex(ValueError, "5x"):
            assess(position_notional_usd="50000.01")
        self.assertEqual(assess(position_notional_usd="50000").gross_leverage,
                         Decimal("5"))


if __name__ == "__main__":
    unittest.main()
