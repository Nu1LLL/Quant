import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))


from config import BacktestConfig, StrategyConfig
from config_io import load_strategy_config, save_strategy_config
from optimizer import (
    OptimizationConfig,
    build_validation_folds,
    evaluate_candidate,
    generate_candidate_configs
)
from rolling_validator import (
    evaluate_rolling_gate,
    summarize_rolling_folds,
    summarize_strategy_folds
)


def make_optimizer_ohlcv(length=800):
    random_generator = np.random.default_rng(20260823)
    time_index = pd.date_range(
        "2020-01-01",
        periods=length,
        freq="4h",
        tz="UTC"
    )
    close = (
        np.linspace(100, 180, length)
        +
        8 * np.sin(np.arange(length) / 15)
        +
        random_generator.normal(0, 0.8, length)
    )
    open_price = np.r_[close[0], close[:-1]]
    high = np.maximum(open_price, close) + 1.0
    low = np.minimum(open_price, close) - 1.0

    return pd.DataFrame(
        {
            "open_time": time_index,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0
        }
    )


class OptimizerTests(unittest.TestCase):
    def test_candidate_generation_is_reproducible(self):
        first = generate_candidate_configs(20, 1234)
        second = generate_candidate_configs(20, 1234)

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 20)

    def test_validation_folds_stop_before_hidden_test(self):
        folds = build_validation_folds(
            data_length=1000,
            development_end=800,
            fold_count=4
        )

        self.assertEqual(folds[-1]["end"], 800)
        self.assertTrue(all(fold["end"] <= 800 for fold in folds))

    def test_hidden_prices_cannot_change_development_score(self):
        original_df = make_optimizer_ohlcv()
        changed_df = original_df.copy()
        hidden_start = int(len(changed_df) * 0.80)
        changed_df.loc[hidden_start:, "close"] *= 5
        changed_df.loc[hidden_start:, "high"] = np.maximum(
            changed_df.loc[hidden_start:, "high"],
            changed_df.loc[hidden_start:, "close"] + 1
        )

        strategy_config = replace(
            StrategyConfig(),
            trend_weight=1.0,
            pullback_weight=0.0,
            ema_window=100
        )
        optimization_config = OptimizationConfig(
            trial_count=2,
            development_fraction=0.80,
            fold_count=2,
            minimum_fold_trades=1
        )

        first_score, first_folds = evaluate_candidate(
            {"TEST": original_df},
            strategy_config,
            BacktestConfig(),
            optimization_config
        )
        second_score, second_folds = evaluate_candidate(
            {"TEST": changed_df},
            strategy_config,
            BacktestConfig(),
            optimization_config
        )

        self.assertEqual(first_score, second_score)
        self.assertEqual(first_folds, second_folds)

    def test_strategy_config_json_round_trip(self):
        strategy_config = replace(
            StrategyConfig(),
            ema_window=150,
            trend_weight=1.0,
            pullback_weight=0.0
        )

        with tempfile.TemporaryDirectory() as temporary_folder:
            file_path = Path(temporary_folder) / "strategy.json"
            save_strategy_config(strategy_config, file_path)
            loaded_config = load_strategy_config(file_path)

        self.assertEqual(strategy_config, loaded_config)

    def test_rolling_summary_and_gate(self):
        fold_df = pd.DataFrame(
            {
                "total_return": [0.03, 0.02, 0.01, -0.01],
                "sharpe_ratio": [1.0, 0.8, 0.6, -0.2],
                "max_drawdown": [-0.05, -0.04, -0.03, -0.08],
                "trade_count": [20, 20, 20, 20]
            }
        )
        summary = summarize_rolling_folds(fold_df)
        strategy_fold_df = pd.DataFrame(
            {
                "strategy": ["trend", "trend", "range", "range"],
                "trade_count": [30, 30, 15, 15],
                "net_profit": [100, 50, 20, 10]
            }
        )
        strategy_summary = summarize_strategy_folds(
            strategy_fold_df
        )
        gate_result = evaluate_rolling_gate(
            summary,
            strategy_summary
        )

        self.assertEqual(summary["total_trades"], 80)
        self.assertTrue(gate_result["passed"])

    def test_rolling_gate_rejects_losing_sub_strategy(self):
        summary = {
            "profitable_fold_ratio": 0.80,
            "median_fold_return": 0.02,
            "worst_fold_return": -0.01,
            "median_sharpe": 1.0,
            "worst_max_drawdown": -0.05,
            "total_trades": 100
        }
        strategy_summary = pd.DataFrame(
            {
                "strategy": ["trend", "range"],
                "trade_count": [80, 20],
                "net_profit": [500.0, -20.0]
            }
        )

        gate_result = evaluate_rolling_gate(
            summary,
            strategy_summary
        )
        self.assertFalse(gate_result["passed"])


if __name__ == "__main__":
    unittest.main()
