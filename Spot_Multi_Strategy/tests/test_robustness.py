import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))


from config import StrategyConfig
from candidate_selector import calculate_candidate_score
from robustness import (
    build_parameter_variants,
    evaluate_robustness_gate,
    run_trade_monte_carlo
)


class RobustnessTests(unittest.TestCase):
    def test_parameter_variants_include_both_directions(self):
        variants = build_parameter_variants(StrategyConfig())

        self.assertIn("ema_minus_20pct", variants)
        self.assertIn("ema_plus_20pct", variants)
        self.assertIn("trailing_minus_20pct", variants)
        self.assertIn("trailing_plus_20pct", variants)
        self.assertEqual(len(variants), 11)

    def test_monte_carlo_is_reproducible(self):
        trades = pd.DataFrame(
            {
                "net_profit": [10.0, -5.0, 20.0, -3.0],
                "entry_account_value": [1000.0, 1010.0, 1005.0, 1025.0]
            }
        )

        first = run_trade_monte_carlo(
            trades,
            initial_capital=1000.0,
            simulation_count=1000,
            random_seed=123
        )
        second = run_trade_monte_carlo(
            trades,
            initial_capital=1000.0,
            simulation_count=1000,
            random_seed=123
        )

        self.assertEqual(first, second)
        self.assertEqual(first["trade_count"], 4)

    def test_gate_rejects_strategy_that_fails_double_costs(self):
        scenario_df = pd.DataFrame(
            {
                "symbol": ["BTC", "ETH", "BTC", "ETH"],
                "scenario": [
                    "all_costs_x2",
                    "all_costs_x2",
                    "delay_2_bars",
                    "delay_2_bars"
                ],
                "total_return": [0.01, -0.01, 0.02, 0.01]
            }
        )
        parameter_df = pd.DataFrame(
            {
                "symbol": ["BTC"] * 10 + ["ETH"] * 10,
                "total_return": [0.01] * 20,
                "sharpe_ratio": [1.0] * 20
            }
        )
        removal_df = pd.DataFrame(
            {
                "symbol": ["BTC", "ETH"],
                "remaining_return": [0.01, 0.01]
            }
        )
        monte_carlo_df = pd.DataFrame(
            {
                "symbol": ["BTC", "ETH"],
                "loss_probability": [0.10, 0.10],
                "percentile_5_max_drawdown": [-0.10, -0.10]
            }
        )

        result = evaluate_robustness_gate(
            scenario_df,
            parameter_df,
            removal_df,
            monte_carlo_df
        )

        self.assertFalse(result["passed"])

    def test_candidate_score_rewards_stability(self):
        stable = {
            "median_sharpe": 1.0,
            "median_fold_return": 0.02,
            "profitable_fold_ratio": 0.80,
            "worst_fold_return": -0.01
        }
        unstable = {
            "median_sharpe": 1.0,
            "median_fold_return": 0.02,
            "profitable_fold_ratio": 0.80,
            "worst_fold_return": -0.20
        }

        self.assertGreater(
            calculate_candidate_score(stable),
            calculate_candidate_score(unstable)
        )


if __name__ == "__main__":
    unittest.main()
