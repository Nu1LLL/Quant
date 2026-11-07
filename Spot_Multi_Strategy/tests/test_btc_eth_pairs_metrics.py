"""Synthetic-only checks for the frozen BTC/ETH pair metrics."""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from research.btc_eth_pairs_metrics import (
    INITIAL_EQUITY,
    OOS_DAYS,
    OOS_START,
    _annual_and_monthly,
    _day_arrays,
    _metrics_from_factors,
    _regime_report,
    evaluate_gate,
    monte_carlo_compressed,
)


def day_record(day, start, minute_marks):
    """Compress a tiny, entirely invented minute path using the pinned rule."""
    peak = maximum = minimum = start
    intraday_dd = 0.0
    for regular, adverse in minute_marks:
        peak = maximum = max(maximum, regular)
        minimum = min(minimum, adverse)
        intraday_dd = max(intraday_dd, 1.0 - adverse / peak)
    return SimpleNamespace(
        date=day,
        equity_start=start,
        equity_end=minute_marks[-1][0],
        max_regular_factor=maximum / start,
        min_adverse_factor=minimum / start,
        max_intraday_drawdown=intraday_dd,
    )


class CompressedDrawdownTests(unittest.TestCase):
    def test_compression_exactly_replays_cross_day_minute_stress(self):
        first = day_record(
            OOS_START,
            10_000.0,
            [(12_000.0, 11_000.0), (14_000.0, 13_500.0), (11_000.0, 10_000.0)],
        )
        second = day_record(
            OOS_START + timedelta(days=1),
            11_000.0,
            [(10_000.0, 8_000.0), (13_000.0, 9_000.0), (10_000.0, 9_000.0)],
        )
        _, factors = _day_arrays([first, second], require_fixed_oos=False)
        replay = _metrics_from_factors(factors)
        self.assertAlmostEqual(replay["max_drawdown"], 1.0 - 8_000.0 / 14_000.0)
        self.assertAlmostEqual(replay["profit_factor"], 1.0)

    def test_invalid_day_continuity_is_rejected(self):
        first = day_record(OOS_START, 10_000.0, [(10_100.0, 10_000.0)])
        bad_second = day_record(OOS_START + timedelta(days=1), 10_200.0, [(10_300.0, 10_200.0)])
        with self.assertRaisesRegex(ValueError, "does not match"):
            _day_arrays([first, bad_second], require_fixed_oos=False)


class MonteCarloTests(unittest.TestCase):
    def test_pcg64_block_replay_is_deterministic_and_undefined_is_counted(self):
        flat = np.tile(np.array([[1.0, 1.0, 1.0, 0.0]]), (30, 1))
        first = monte_carlo_compressed(flat, paths=37, block_days=7, seed=20261108)
        second = monte_carlo_compressed(flat, paths=37, block_days=7, seed=20261108)
        self.assertEqual(first, second)
        self.assertEqual(first["joint_pass_count"], 0)
        self.assertEqual(first["undefined_counts"]["sharpe"], 37)
        self.assertEqual(first["undefined_counts"]["profit_factor"], 37)
        self.assertEqual(first["percentiles"]["max_drawdown"]["p95"], 0.0)


class GateTests(unittest.TestCase):
    @staticmethod
    def synthetic_scenario():
        records = []
        equity = INITIAL_EQUITY
        for j in range(OOS_DAYS):
            factor = 1.003 if j % 2 == 0 else 0.9995
            end = equity * factor
            records.append(day_record(OOS_START + timedelta(days=j), equity, [(end, min(equity, end))]))
            equity = end
        _, factors = _day_arrays(records, require_fixed_oos=True)
        trades = []
        regimes = ("bull_high", "bull_low", "bear_high", "bear_low", "bull_high")
        for year, regime in zip(range(2021, 2026), regimes):
            signal = datetime(year, 6, 1, 1, tzinfo=timezone.utc)
            fill = signal + timedelta(hours=1, minutes=1)
            trades.append(SimpleNamespace(
                entry_signal_time=signal,
                entry_fill_time=fill,
                exit_fill_time=fill + timedelta(hours=5),
                direction=1,
                regime=regime,
                net_pnl=100.0,
            ))
        return SimpleNamespace(
            daily_records=records,
            trades=trades,
            max_drawdown=_metrics_from_factors(factors)["max_drawdown"],
        )

    def test_annual_monthly_and_four_regimes(self):
        scenario = self.synthetic_scenario()
        annual, monthly, summary = _annual_and_monthly(scenario.daily_records, scenario.trades)
        self.assertEqual(len(annual), 5)
        self.assertTrue(all(fold["pass"] for fold in annual.values()))
        self.assertEqual(len(monthly), 60)
        self.assertEqual(summary["positive"], 60)
        buckets, passed = _regime_report(scenario.trades)
        self.assertTrue(passed)
        self.assertEqual(sum(buckets[key]["trades"] for key in buckets), 5)

    def test_full_gate_keeps_independent_execution_evidence_unpassed(self):
        scenario = self.synthetic_scenario()
        scenarios = {
            name: scenario for name in (
                "base_1x", "base_2x", "doubled_cost_2x", "no_funding_credit_2x"
            )
        }
        with patch("research.btc_eth_pairs_metrics.monte_carlo_compressed", return_value={
            "joint_pass_fraction": 1.0,
        }):
            report = evaluate_gate(scenarios)
        self.assertTrue(report["gate"]["all_numeric"])
        self.assertFalse(report["gate"]["full"])
        self.assertTrue(report["formal_cagr_within_40_to_60_percent_target_band"])
        self.assertEqual(report["completed_pair_round_trips"], 5)


if __name__ == "__main__":
    unittest.main()
