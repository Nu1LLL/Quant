import unittest

import numpy as np
import pandas as pd

from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
    cross_sectional_ic_series,
)


class StabilitySignificanceTests(unittest.TestCase):
    def test_bootstrap_is_deterministic(self):
        values = np.tile([0.01, -0.005, 0.02, 0.0], 100)
        first = circular_block_bootstrap_mean_pvalue(
            values, block_length=4, repetitions=199, seed=7
        )
        second = circular_block_bootstrap_mean_pvalue(
            values, block_length=4, repetitions=199, seed=7
        )
        self.assertEqual(first, second)

    def test_positive_constant_has_small_pvalue(self):
        value = circular_block_bootstrap_mean_pvalue(
            np.ones(200), block_length=10, repetitions=199, seed=7
        )
        self.assertEqual(value, 1 / 200)

    def test_benjamini_hochberg_known_example_and_nan(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
        np.testing.assert_allclose(adjusted[:3], [0.03, 0.04, 0.04])
        self.assertTrue(np.isnan(adjusted[3]))

    def test_cross_sectional_ic_preserves_timestamps(self):
        index = pd.date_range("2024-01-01", periods=2, freq="D")
        signals = pd.DataFrame(
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], index=index
        )
        forwards = pd.DataFrame(
            [[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]], index=index
        )
        result = cross_sectional_ic_series(signals, forwards)
        np.testing.assert_allclose(result.to_numpy(), [1.0, -1.0])
        self.assertTrue(result.index.equals(index))


if __name__ == "__main__":
    unittest.main()
