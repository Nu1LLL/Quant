import unittest

import numpy as np

import bocpd


class BocpdTests(unittest.TestCase):
    def test_mean_shift_raises_online_changepoint_probability(self):
        rng = np.random.default_rng(4)
        train = rng.normal(0, 1, 200)
        observations = np.r_[rng.normal(0, 1, 100), rng.normal(5, 1, 20)]
        probability = bocpd.changepoint_probabilities(
            observations, bocpd.prior_from_training(train)
        )
        self.assertGreater(probability[100:105].max(), probability[50:90].max())

    def test_appending_future_does_not_rewrite_changepoint_history(self):
        rng = np.random.default_rng(5)
        train = rng.normal(size=100)
        values = rng.normal(size=150)
        prior = bocpd.prior_from_training(train)
        prefix = bocpd.changepoint_probabilities(values[:80], prior)
        extended = bocpd.changepoint_probabilities(values, prior)
        np.testing.assert_allclose(prefix, extended[:80])

    def test_probability_union_is_bounded(self):
        matrix = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
        combined = bocpd.combine_probabilities(matrix)
        np.testing.assert_allclose(combined, [0.496, 0.994])
        self.assertTrue(((combined >= 0) & (combined <= 1)).all())
