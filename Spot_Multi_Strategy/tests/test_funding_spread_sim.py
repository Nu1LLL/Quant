from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from research.funding_spread_data import FundingEvent, HourBar, SYMBOLS, TerminalOpen
from research.funding_spread_sim import (
    END, INPUT_START, START, _Scenario, run_funding_spread_simulation,
)
from research.funding_spread_metrics import evaluate_funding_spread_gate


UTC = timezone.utc


class FundingSpreadSimulationTests(unittest.TestCase):
    def test_full_calendar_synthetic_signal_uses_only_prior_settlements(self):
        bars = []
        time = INPUT_START
        while time < END:
            bars.append(HourBar(time, (100.0,) * 3, (100.0,) * 3,
                                (100.0,) * 3, (100.0,) * 3))
            time += timedelta(hours=1)
        bars.append(TerminalOpen(END, (100.0,) * 3))
        events = []
        anchor = datetime(2021, 1, 1, tzinfo=UTC)
        while anchor < END:
            for milliseconds, symbol, rate in ((1, "DOGEUSDT", -.001),
                                               (3, "ADAUSDT", .001),
                                               (5, "BNBUSDT", 0.0)):
                events.append(FundingEvent(anchor + timedelta(milliseconds=milliseconds),
                                           symbol, rate))
            anchor += timedelta(hours=8)
        scenarios = run_funding_spread_simulation(bars, events)
        for scenario in scenarios.values():
            self.assertEqual(scenario.decision_opportunities, 59)
            self.assertEqual(len(scenario.trades), 59)
            self.assertEqual(len(scenario.daily_records), 1826)
            self.assertEqual(scenario.daily_records[0].date, START.date())
            self.assertEqual(scenario.daily_records[-1].date,
                             datetime(2025, 12, 31, tzinfo=UTC).date())
            self.assertEqual(scenario.trades[0].entry_fill_time,
                             datetime(2021, 2, 1, 1, tzinfo=UTC))
            self.assertEqual(scenario.trades[-1].exit_fill_time, END)
            self.assertTrue(all(t.long_symbol == "DOGEUSDT"
                                and t.short_symbol == "ADAUSDT"
                                for t in scenario.trades))
            self.assertEqual(scenario.trades[0].regime, "bull_high")
            self.assertEqual(scenario.daily_records[0].equity_end, 10_000.0)
        self.assertGreater(scenarios["base_2x"].final_equity,
                           scenarios["base_1x"].final_equity)
        self.assertLess(scenarios["doubled_cost_2x"].final_equity,
                        scenarios["base_2x"].final_equity)
        self.assertLess(scenarios["no_funding_credit_2x"].final_equity,
                        scenarios["base_2x"].final_equity)
        with patch("research.funding_spread_metrics.monte_carlo_compressed",
                   return_value={"joint_pass_fraction": 0.0}) as monte_carlo:
            report = evaluate_funding_spread_gate(scenarios)
        monte_carlo.assert_called_once()
        self.assertTrue(all(report["compressed_replay_verified"].values()))
        self.assertEqual(len(report["monthly_returns"]), 60)
        self.assertEqual(len(report["annual_folds"]), 5)
        self.assertEqual(report["completed_pairs"], 59)
        self.assertFalse(report["gate"]["full"])

    def test_hourly_long_low_and_short_high_extrema_are_jointly_adverse(self):
        t = datetime(2021, 2, 1, 1, tzinfo=UTC)
        account = _Scenario("test", 2.0, .0025, False)
        account.boundary((100.0,) * 3)
        account.start_day(t.date())
        account.enter(t, (100.0,) * 3, t - timedelta(hours=8),
                      SYMBOLS.index("DOGEUSDT"), SYMBOLS.index("ADAUSDT"),
                      t + timedelta(days=28), "bull_high")
        bar = HourBar(t, (100.0,) * 3, (120.0, 100.0, 110.0),
                      (90.0, 100.0, 80.0), (110.0, 100.0, 110.0))
        account.mark(bar)
        self.assertGreater(account.max_drawdown, .39)
        self.assertGreater(account.last_regular, 0)


if __name__ == "__main__":
    unittest.main()
