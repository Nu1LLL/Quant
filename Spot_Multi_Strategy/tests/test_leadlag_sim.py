from __future__ import annotations

from datetime import date, datetime, timezone
import unittest
from unittest.mock import patch

import numpy as np

from research.leadlag_data import PriceTape, SYMBOLS
from research.leadlag_metrics import evaluate_leadlag_gate
from research.leadlag_sim import (
    BASE, EXPECTED_BARS, _Account, _monthly_model, _rolling_volatility,
    run_leadlag_simulation,
)


class LeadLagSimulationTests(unittest.TestCase):
    def test_monthly_model_uses_only_embargoed_past_labels(self):
        rng = np.random.default_rng(617)
        cutoff = 30 * 288 + 8
        opening = 100 * np.exp(np.cumsum(rng.normal(0, .001, (cutoff + 30, 10)), axis=0))
        returns = np.empty_like(opening)
        returns[0] = np.nan
        returns[1:] = np.log(opening[1:] / opening[:-1])
        first = _monthly_model(opening, returns, cutoff)
        altered = opening.copy()
        altered[cutoff:] *= 2.0
        second = _monthly_model(altered, returns, cutoff)
        self.assertGreaterEqual(first.training_rows, 8000)
        np.testing.assert_array_equal(first.coefficients, second.coefficients)
        np.testing.assert_array_equal(first.intercept, second.intercept)
        self.assertTrue(np.allclose(np.diag(first.coefficients), 0.0))

    def test_adverse_five_minute_mark_and_costs(self):
        opening = np.full((4, len(SYMBOLS)), 100.0)
        high = np.full_like(opening, 110.0)
        low = np.full_like(opening, 90.0)
        close = opening.copy()
        tape = PriceTape(opening, high, low, close, 0)
        account = _Account("synthetic", 2.0, .0005, 1)
        account.start_day(date(2021, 1, 1))
        account.execute(tape, 0, 0, 1, "bull_high", .005)
        account.finish_day()
        self.assertEqual(len(account.trades), 1)
        self.assertLess(account.cash, 10_000.0)
        self.assertGreater(account.max_drawdown, .19)
        self.assertEqual(account.trades[0].funding_pnl, 0)
        self.assertAlmostEqual(account.trades[0].net_pnl,
                               account.trades[0].gross_price_pnl
                               + account.trades[0].fee_pnl)

    def test_first_valid_daily_volatility_snapshot_uses_full_288_returns(self):
        values = np.full((300, 10), .001)
        values[0] = np.nan
        result = _rolling_volatility(values)
        self.assertTrue(np.isnan(result[287]))
        self.assertTrue(np.isfinite(result[288]))

    def test_five_year_synthetic_calendar_and_gate_replay(self):
        steps = np.arange(EXPECTED_BARS, dtype=np.float64)[:, None]
        coins = np.arange(len(SYMBOLS), dtype=np.float64)[None, :]
        opening = 100 * np.exp(.0000002 * steps
                               + .001 * np.sin(steps / (17 + coins * 3))
                               + .0005 * np.sin(steps / (7 + coins)))
        tape = PriceTape(opening, opening * 1.001, opening * .999,
                         opening.copy(), int(BASE.timestamp() * 1000))
        scenarios = run_leadlag_simulation(tape)
        for scenario in scenarios.values():
            self.assertEqual(len(scenario.daily_records), 1826)
            self.assertEqual(scenario.decision_opportunities, 72 * 1826)
            self.assertEqual(scenario.monthly_fits, 60)
            self.assertEqual(scenario.min_training_rows, 8637)
            self.assertEqual(len(scenario.trades), 0)
        with patch("research.leadlag_metrics.monte_carlo_compressed",
                   return_value={"joint_pass_fraction": 0.0}):
            report = evaluate_leadlag_gate(scenarios, funding_schedule_verified=True)
        self.assertTrue(all(report["compressed_replay_verified"].values()))
        self.assertEqual(len(report["monthly_returns"]), 60)
        self.assertFalse(report["gate"]["full"])

        class OneSignalModel:
            training_rows = 8637
            calls = 0

            def predict(self, unused):
                self.calls += 1
                values = np.zeros(len(SYMBOLS))
                if self.calls == 1:
                    values[0] = .005
                return values

        one_signal = OneSignalModel()
        with patch("research.leadlag_sim._monthly_model", return_value=one_signal):
            traded = run_leadlag_simulation(tape)
        for scenario in traded.values():
            self.assertEqual(len(scenario.trades), 1)
            trade = scenario.trades[0]
            self.assertEqual(trade.entry_signal_time,
                             datetime(2021, 1, 1, tzinfo=timezone.utc))
            self.assertEqual(trade.exit_fill_time,
                             datetime(2021, 1, 1, 0, 15, tzinfo=timezone.utc))
            expected_entry_minute = 10 if scenario is traded["latency_2x"] else 5
            self.assertEqual(trade.entry_fill_time.minute, expected_entry_minute)


if __name__ == "__main__":
    unittest.main()
