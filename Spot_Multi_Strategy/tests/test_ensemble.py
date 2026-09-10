import unittest

import numpy as np
import pandas as pd

import alphas
import ensemble


def _make_ohlcv(rows, seed, start_price=100.0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0003, scale=0.008, size=rows)
    close = start_price * np.cumprod(1 + returns)

    open_ = np.empty(rows)
    open_[0] = start_price
    open_[1:] = close[:-1]

    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, rows))
    volume = rng.uniform(100, 1000, rows)

    open_time = pd.date_range(
        "2020-01-01", periods=rows, freq="4h", tz="UTC"
    )

    return pd.DataFrame({
        "open_time": open_time,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })


class EnsembleWeightingTests(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(rows=3000, seed=5)
        all_alphas = alphas.build_single_asset_alphas(self.df)
        self.alpha_signals = {
            name: all_alphas[name]
            for name in [
                "A01_ts_momentum_24",
                "A02_ema_distance_50",
                "A06_rsi_reversion_14"
            ]
        }

    def test_equal_weights_sum_to_one_where_data_available(self):
        signal_matrix = ensemble.normalized_signal_matrix(self.alpha_signals)
        weights = ensemble.equal_weights(signal_matrix)

        fully_available_rows = signal_matrix.notna().all(axis=1)
        row_sums = weights.loc[fully_available_rows].sum(axis=1)

        self.assertTrue(
            np.allclose(row_sums, 1.0, atol=1e-9)
        )

    def test_ic_weights_sum_to_one_or_fallback(self):
        weights = ensemble.ic_weights(
            self.alpha_signals, self.df, horizon=3, window=300, min_periods=50
        )
        signal_matrix = ensemble.normalized_signal_matrix(self.alpha_signals)
        fully_available_rows = signal_matrix.notna().all(axis=1)

        row_sums = weights.loc[fully_available_rows].sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6))

    def test_ic_weights_do_not_use_unrealized_future_information(self):
        weights_before = ensemble.ic_weights(
            self.alpha_signals, self.df, horizon=3, window=300, min_periods=50
        )

        mutated_df = self.df.copy()
        cutoff = int(len(mutated_df) * 0.9)
        mutated_df.loc[cutoff:, ["open", "high", "low", "close"]] *= 2.0
        mutated_alphas = alphas.build_single_asset_alphas(mutated_df)
        mutated_signals = {
            name: mutated_alphas[name] for name in self.alpha_signals
        }

        weights_after = ensemble.ic_weights(
            mutated_signals, mutated_df,
            horizon=3, window=300, min_periods=50
        )

        # 早期的权重（远早于被篡改的区间加上horizon+delay的滞后）
        # 不应该因为未来数据被改动而改变
        safe_boundary = cutoff - 50
        pd.testing.assert_frame_equal(
            weights_before.iloc[:safe_boundary],
            weights_after.iloc[:safe_boundary]
        )

    def test_correlation_penalty_reduces_weight_of_redundant_alpha(self):
        rng = np.random.default_rng(3)
        base_signal = pd.Series(rng.normal(size=len(self.df)))
        near_duplicate = base_signal + rng.normal(scale=0.01, size=len(self.df))
        independent = pd.Series(rng.normal(size=len(self.df)))

        redundant_matrix = pd.DataFrame({
            "alpha_a": base_signal.clip(-1, 1),
            "alpha_b_duplicate": near_duplicate.clip(-1, 1),
            "alpha_c_independent": independent.clip(-1, 1)
        })
        base_weights = pd.DataFrame(
            1 / 3, index=redundant_matrix.index, columns=redundant_matrix.columns
        )

        penalized = ensemble.correlation_penalized_weights(
            base_weights, redundant_matrix, window=200, min_periods=50
        )

        tail = penalized.iloc[-200:]
        self.assertLess(
            tail["alpha_a"].mean(), tail["alpha_c_independent"].mean()
        )
        self.assertLess(
            tail["alpha_b_duplicate"].mean(), tail["alpha_c_independent"].mean()
        )

    def test_target_exposure_is_bounded_between_zero_and_one(self):
        for method in ("equal", "ic", "corr_penalized"):
            _, _, exposure = ensemble.build_ensemble(
                self.alpha_signals, self.df, method=method,
                horizon=3, window=300, min_periods=50
            )
            valid = exposure.dropna()
            self.assertTrue((valid >= 0.0).all())
            self.assertTrue((valid <= 1.0).all())

    def test_negative_combined_alpha_maps_to_zero_exposure(self):
        combined = pd.Series([-0.8, -0.1, 0.0, 0.3, 1.5])
        exposure = ensemble.target_exposure_from_combined_alpha(combined)
        self.assertListEqual(list(exposure), [0.0, 0.0, 0.0, 0.3, 1.0])


if __name__ == "__main__":
    unittest.main()
