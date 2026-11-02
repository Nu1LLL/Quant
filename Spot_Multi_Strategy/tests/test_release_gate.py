import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))


from release_gate import calculate_paper_metrics, evaluate_release_gate


def make_state():
    return {
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_run_at": "2026-04-01T00:00:00+00:00",
        "config_hash": "correct",
        "data_gap_count": 0,
        "accounts": {
            "BTCUSDT": {
                "initial_capital": 500.0,
                "closed_trades": 6
            },
            "ETHUSDT": {
                "initial_capital": 500.0,
                "closed_trades": 6
            }
        }
    }


class ReleaseGateTests(unittest.TestCase):
    def test_complete_paper_record_can_pass(self):
        state = make_state()
        events_df = pd.DataFrame(
            {"net_profit": [20.0] * 8 + [-10.0] * 4}
        )
        equity_df = pd.DataFrame(
            {
                "run_time": [
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                    "2026-04-01T00:00:00+00:00",
                    "2026-04-01T00:00:00+00:00"
                ],
                "equity": [500, 500, 550, 550]
            }
        )
        metrics = calculate_paper_metrics(
            state,
            events_df,
            equity_df,
            now="2026-04-01T01:00:00+00:00"
        )
        checks, passed = evaluate_release_gate(
            {"gate_passed": True},
            {"gate_passed": True},
            state,
            metrics,
            expected_config_hash="correct"
        )

        self.assertTrue(passed)
        self.assertTrue(all(checks.values()))
        self.assertEqual(metrics["profit_factor"], 4.0)

    def test_recent_but_short_record_is_rejected(self):
        state = make_state()
        state["created_at"] = "2026-03-20T00:00:00+00:00"
        metrics = {
            "paper_days": 12,
            "state_age_hours": 1,
            "closed_trade_count": 12,
            "profit_factor": 2,
            "maximum_drawdown": 0.05,
            "net_profit": 100
        }
        checks, passed = evaluate_release_gate(
            {"gate_passed": True},
            {"gate_passed": True},
            state,
            metrics,
            expected_config_hash="correct"
        )

        self.assertFalse(passed)
        self.assertFalse(checks["连续模拟不少于90天"])

    def test_drawdown_includes_starting_capital(self):
        state = make_state()
        equity_df = pd.DataFrame(
            {
                "run_time": ["2026-04-01T00:00:00+00:00"],
                "equity": [900.0]
            }
        )
        metrics = calculate_paper_metrics(
            state,
            pd.DataFrame(),
            equity_df,
            now="2026-04-01T01:00:00+00:00"
        )

        self.assertAlmostEqual(metrics["maximum_drawdown"], 0.10)


if __name__ == "__main__":
    unittest.main()
