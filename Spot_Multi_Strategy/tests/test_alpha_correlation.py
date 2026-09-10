import unittest

import numpy as np
import pandas as pd

import alpha_correlation


class AlphaCorrelationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 500
        base = rng.normal(size=n)

        self.signal_matrix = pd.DataFrame({
            "alpha_original": base,
            "alpha_duplicate": base + rng.normal(scale=0.01, size=n),
            "alpha_independent": rng.normal(size=n),
        })

    def test_pearson_matrix_flags_near_duplicate(self):
        matrix = alpha_correlation.pearson_correlation_matrix(
            self.signal_matrix
        )
        self.assertGreater(
            matrix.loc["alpha_original", "alpha_duplicate"], 0.99
        )
        self.assertLess(
            abs(matrix.loc["alpha_original", "alpha_independent"]), 0.2
        )

    def test_find_redundant_pairs_respects_threshold(self):
        matrix = alpha_correlation.pearson_correlation_matrix(
            self.signal_matrix
        )
        redundant = alpha_correlation.find_redundant_pairs(
            matrix, threshold=0.75
        )
        pair_names = set(
            frozenset([row.alpha_a, row.alpha_b])
            for row in redundant.itertuples(index=False)
        )
        self.assertIn(
            frozenset(["alpha_original", "alpha_duplicate"]), pair_names
        )
        self.assertNotIn(
            frozenset(["alpha_original", "alpha_independent"]), pair_names
        )

    def test_effective_independent_signals_is_between_one_and_count(self):
        matrix = alpha_correlation.pearson_correlation_matrix(
            self.signal_matrix
        )
        n_eff = alpha_correlation.effective_independent_signals(matrix)
        self.assertGreaterEqual(n_eff, 1.0)
        self.assertLessEqual(n_eff, len(self.signal_matrix.columns) + 1e-6)

    def test_effective_independent_signals_drops_with_duplication(self):
        independent_only = self.signal_matrix[
            ["alpha_original", "alpha_independent"]
        ]
        duplicated = self.signal_matrix[
            ["alpha_original", "alpha_duplicate", "alpha_independent"]
        ]

        n_eff_independent = alpha_correlation.effective_independent_signals(
            alpha_correlation.pearson_correlation_matrix(independent_only)
        )
        n_eff_duplicated = alpha_correlation.effective_independent_signals(
            alpha_correlation.pearson_correlation_matrix(duplicated)
        )

        # 加入一个几乎重复的信号后，有效独立信号数量增量应该远小于1
        self.assertLess(
            n_eff_duplicated - n_eff_independent, 0.5
        )

    def test_spearman_matrix_does_not_require_scipy(self):
        matrix = alpha_correlation.spearman_correlation_matrix(
            self.signal_matrix
        )
        self.assertGreater(
            matrix.loc["alpha_original", "alpha_duplicate"], 0.9
        )


if __name__ == "__main__":
    unittest.main()
