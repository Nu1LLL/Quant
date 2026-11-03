"""Synthetic, no-market-data tests for research-only crypto archetypes."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest

from research.crypto_blackbox.archetypes import (
    Candidate,
    FlowConfig,
    FlowSnapshot,
    ObservedValue,
    RisingConfig,
    RisingSnapshot,
    flow_candidate,
    rising_tide_candidates,
)


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def seen(value, minutes_ago=0):
    return ObservedValue(value, NOW - timedelta(minutes=minutes_ago))


def flow(*, symbol="BTCUSDT", market="crypto_perpetual", deviation=-80.0,
         ofi=0.6, spread=1.0, volatility=80.0, liquidity=2_000_000.0):
    return FlowSnapshot(
        symbol=symbol, venue="SYNTHETIC", market=market,
        ofi=seen(ofi), spread_bps=seen(spread),
        anchor_deviation_bps=seen(deviation), volatility_bps=seen(volatility),
        liquidity_usd=seen(liquidity),
    )


def rising(symbol, theme, strength, *, market="crypto_perpetual", beta=2.0,
           spread=2.0, liquidity=2_000_000.0, volatility=100.0, stress=0.1):
    return RisingSnapshot(
        symbol=symbol, venue="SYNTHETIC", market=market,
        theme=theme, theme_available_at=NOW - timedelta(days=30),
        strength=seen(strength), beta=seen(beta), spread_bps=seen(spread),
        liquidity_usd=seen(liquidity), volatility_bps=seen(volatility),
        stress_score=seen(stress),
    )


class FlowArchetypeTests(unittest.TestCase):
    def setUp(self):
        self.config = FlowConfig(
            reversion_capture_fraction=0.50,
            roundtrip_fee_bps=2.0,
            roundtrip_slippage_bps=1.0,
            funding_buffer_bps=0.0,
        )

    def test_long_short_and_no_trade_are_non_deployable(self):
        long = flow_candidate(flow(), NOW, self.config)
        short = flow_candidate(flow(deviation=80.0, ofi=-0.6), NOW, self.config)
        flat = flow_candidate(flow(deviation=80.0, ofi=0.6), NOW, self.config)
        self.assertEqual((long.side, short.side, flat.side),
                         ("LONG", "SHORT", "FLAT"))
        self.assertGreater(long.target_weight, 0.0)
        self.assertLess(short.target_weight, 0.0)
        self.assertEqual(flat.target_weight, 0.0)
        for candidate in (long, short, flat):
            self.assertEqual(candidate.market, "crypto_perpetual")
            self.assertTrue(candidate.research_only)
            self.assertFalse(candidate.deployable)
            self.assertEqual(candidate.horizon_minutes, 60)
            self.assertTrue(candidate.reason)

    def test_cost_or_risk_deterioration_cannot_create_a_trade(self):
        original = flow_candidate(flow(), NOW, self.config)
        expensive = flow_candidate(
            flow(), NOW, replace(self.config, roundtrip_fee_bps=100.0)
        )
        illiquid = flow_candidate(flow(liquidity=100.0), NOW, self.config)
        stressed = flow_candidate(flow(volatility=1000.0), NOW, self.config)
        self.assertEqual(original.side, "LONG")
        self.assertEqual([expensive.side, illiquid.side, stressed.side],
                         ["FLAT", "FLAT", "FLAT"])
        self.assertEqual(expensive.reason, "assumed_edge_does_not_clear_costs")

    def test_no_unsupported_spot_short_options_or_equity(self):
        spot_short = flow_candidate(
            flow(market="crypto_spot", deviation=80.0, ofi=-0.6), NOW, self.config
        )
        spot_long = flow_candidate(flow(market="crypto_spot"), NOW, self.config)
        option = flow_candidate(flow(market="crypto_options"), NOW, self.config)
        self.assertEqual(spot_short.side, "FLAT")
        self.assertEqual(spot_long.side, "LONG")
        self.assertEqual(option.side, "FLAT")
        with self.assertRaisesRegex(ValueError, "only crypto"):
            flow_candidate(flow(market="equity"), NOW, self.config)

    def test_future_or_stale_features_fail_closed(self):
        future = replace(flow(), ofi=ObservedValue(0.6, NOW + timedelta(seconds=1)))
        stale = replace(flow(), spread_bps=seen(1.0, minutes_ago=6))
        with self.assertRaisesRegex(ValueError, "not available"):
            flow_candidate(future, NOW, self.config)
        with self.assertRaisesRegex(ValueError, "stale"):
            flow_candidate(stale, NOW, self.config)
        with self.assertRaisesRegex(ValueError, "UTC"):
            flow_candidate(flow(), NOW.replace(tzinfo=None), self.config)


class RisingTideArchetypeTests(unittest.TestCase):
    def test_theme_and_total_budgets_layers_ranking_and_unique_symbols(self):
        rows = [
            rising("CCCUSDT", "meme", 0.90),
            rising("DDDUSDT", "layer1", 0.85, market="crypto_spot"),
            rising("AAAUSDT", "meme", 0.95),
            rising("EEEUSDT", "infra", 0.70),
            rising("BBBUSDT", "meme", 0.92),
        ]
        candidates = rising_tide_candidates(rows, NOW)
        by_symbol = {item.symbol: item for item in candidates}
        self.assertEqual(len(by_symbol), len(rows))
        self.assertEqual(by_symbol["AAAUSDT"].side, "LONG")
        self.assertEqual(by_symbol["AAAUSDT"].entry_layers, 3)
        self.assertAlmostEqual(by_symbol["AAAUSDT"].target_weight, 0.20)
        self.assertAlmostEqual(by_symbol["BBBUSDT"].target_weight, 0.10)
        self.assertEqual(by_symbol["CCCUSDT"].side, "FLAT")
        self.assertEqual(by_symbol["CCCUSDT"].reason, "theme_asset_limit")
        self.assertEqual(by_symbol["DDDUSDT"].entry_layers, 2)
        self.assertEqual(by_symbol["DDDUSDT"].market, "crypto_spot")
        self.assertEqual(by_symbol["EEEUSDT"].entry_layers, 1)
        self.assertLessEqual(sum(c.target_weight for c in candidates), 0.60 + 1e-12)
        self.assertLessEqual(sum(c.target_weight for c in candidates
                                 if c.theme == "meme"), 0.30 + 1e-12)
        self.assertTrue(all(c.target_weight >= 0.0 and not c.deployable
                            and c.research_only for c in candidates))
        reverse = rising_tide_candidates(reversed(rows), NOW)
        self.assertEqual(
            {c.symbol: c.target_weight for c in candidates},
            {c.symbol: c.target_weight for c in reverse},
        )

    def test_point_in_time_risk_and_data_quality_gates(self):
        rows = [
            rising("LOWUSDT", "theme", 0.60),
            rising("STRESSUSDT", "theme", 0.95, stress=0.9),
            rising("ILLIQUSDT", "theme", 0.95, liquidity=1.0),
            rising("WIDEUSDT", "theme", 0.95, spread=100.0),
            rising("OPTIONUSDT", "theme", 0.95, market="crypto_options"),
        ]
        candidates = rising_tide_candidates(rows, NOW)
        self.assertTrue(all(candidate.side == "FLAT" for candidate in candidates))
        self.assertTrue(all(candidate.target_weight == 0.0 for candidate in candidates))

    def test_future_theme_or_feature_and_duplicates_fail_closed(self):
        row = rising("AAAUSDT", "meme", 0.95)
        with self.assertRaisesRegex(ValueError, "theme is not available"):
            rising_tide_candidates([replace(row, theme_available_at=NOW +
                                             timedelta(seconds=1))], NOW)
        with self.assertRaisesRegex(ValueError, "strength is not available"):
            rising_tide_candidates([replace(row, strength=ObservedValue(
                0.95, NOW + timedelta(seconds=1)))], NOW)
        with self.assertRaisesRegex(ValueError, "duplicate crypto symbol"):
            rising_tide_candidates([row, replace(row, symbol="aaausdt")], NOW)
        with self.assertRaisesRegex(ValueError, "only crypto"):
            rising_tide_candidates([replace(row, market="equity")], NOW)

    def test_config_rejects_noncausal_or_overlevered_settings(self):
        with self.assertRaises(ValueError):
            RisingConfig(max_total_weight=1.2)
        with self.assertRaises(ValueError):
            RisingConfig(entry_layer_fractions=(0.5, 0.5, 0.5))
        with self.assertRaises(ValueError):
            FlowConfig(max_abs_weight=2.0)
        with self.assertRaises(ValueError):
            ObservedValue(float("nan"), NOW)
        with self.assertRaisesRegex(TypeError, "deployable"):
            Candidate("flow", "crypto_perpetual", "SYNTHETIC", "BTCUSDT",
                      "LONG", 0.1, 60, "synthetic", deployable=True)
        with self.assertRaisesRegex(ValueError, "target_weight"):
            Candidate("flow", "crypto_perpetual", "SYNTHETIC", "BTCUSDT",
                      "LONG", 2.0, 60, "synthetic")


if __name__ == "__main__":
    unittest.main()
