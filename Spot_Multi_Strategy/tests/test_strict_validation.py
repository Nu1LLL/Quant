import unittest

import numpy as np
import pandas as pd

import strict_validation


class StrictValidationTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-01-01", periods=1512, freq="B", tz="UTC")
        rng = np.random.default_rng(10)
        self.returns = pd.Series(0.0015 + rng.normal(0, 0.004, len(self.index)), index=self.index)

    def test_seen_history_cannot_pass_as_independent_oos(self):
        result = strict_validation.evaluate_strict_oos(
            self.returns, independent_oos=False, costs_included=True
        )
        self.assertFalse(result["checks"]["independent_oos"])
        self.assertFalse(result["passed"])

    def test_costs_must_be_included(self):
        result = strict_validation.evaluate_strict_oos(
            self.returns, independent_oos=True, costs_included=False
        )
        self.assertFalse(result["checks"]["costs_included"])
        self.assertFalse(result["passed"])

    def test_walk_forward_has_complete_annual_folds(self):
        folds, passed = strict_validation.annual_walk_forward(self.returns)
        self.assertGreaterEqual(len(folds), 5)
        self.assertIn("fold_pass", folds)
        self.assertIsInstance(passed, bool)

    def test_monte_carlo_is_reproducible_and_reports_joint_probability(self):
        first = strict_validation.circular_block_monte_carlo(
            self.returns, simulations=100, seed=123
        )
        second = strict_validation.circular_block_monte_carlo(
            self.returns, simulations=100, seed=123
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["joint_target_probability"], 0.0)
        self.assertLessEqual(first["joint_target_probability"], 1.0)

    def test_regime_metrics_keep_named_groups(self):
        regimes = pd.Series(
            np.where(np.arange(len(self.index)) % 2, "stress", "normal"),
            index=self.index
        )
        result = strict_validation.regime_metrics(self.returns, regimes)
        self.assertEqual(set(result["regime"]), {"normal", "stress"})


if __name__ == "__main__":
    unittest.main()
