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


class RegimeConditionalEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(rows=2000, seed=9)
        all_alphas = alphas.build_single_asset_alphas(self.df)
        self.alpha_signals = {
            name: all_alphas[name]
            for name in [
                "A01_ts_momentum_24",
                "A02_ema_distance_50",
                "A06_rsi_reversion_14"
            ]
        }
        rng = np.random.default_rng(21)
        block_size = 40
        block_count = len(self.df) // block_size + 1
        block_regime = rng.choice(
            ["trending", "ranging"], size=block_count
        )
        self.regime_labels = pd.Series(
            np.repeat(block_regime, block_size)[:len(self.df)]
        )

    def test_alpha_only_gets_weight_in_its_licensed_regime(self):
        regime_alpha_map = {
            "trending": ["A02_ema_distance_50"],
            "ranging": ["A06_rsi_reversion_14"],
            "mixed": []
        }
        weights, _, _ = ensemble.build_regime_conditional_ensemble(
            self.alpha_signals, self.df, self.regime_labels,
            regime_alpha_map, method="equal"
        )

        # 跳过最前面的warmup区间：A02/A06都有几十根K线的滚动窗口，
        # 在warmup期间normalized_signal本身是NaN，等权权重合法地是0，
        # 这不代表regime licensing出了问题。
        warmup_buffer = 200
        trending_rows = (self.regime_labels == "trending") & (
            self.regime_labels.index >= warmup_buffer
        )
        ranging_rows = (self.regime_labels == "ranging") & (
            self.regime_labels.index >= warmup_buffer
        )

        self.assertTrue(
            (weights.loc[ranging_rows, "A02_ema_distance_50"] == 0.0).all()
        )
        self.assertTrue(
            (weights.loc[trending_rows, "A06_rsi_reversion_14"] == 0.0).all()
        )
        self.assertTrue(
            (weights.loc[trending_rows, "A02_ema_distance_50"] > 0.0).all()
        )
        self.assertTrue(
            (weights.loc[ranging_rows, "A06_rsi_reversion_14"] > 0.0).all()
        )
        # A01从来没有被任何regime license，权重应该恒为0
        self.assertTrue((weights["A01_ts_momentum_24"] == 0.0).all())

    def test_unlicensed_regime_produces_zero_exposure(self):
        regime_alpha_map = {
            "trending": ["A02_ema_distance_50"],
            "ranging": [],
            "mixed": []
        }
        weights, combined_alpha, exposure = (
            ensemble.build_regime_conditional_ensemble(
                self.alpha_signals, self.df, self.regime_labels,
                regime_alpha_map, method="equal"
            )
        )
        ranging_rows = self.regime_labels == "ranging"
        self.assertTrue((weights.loc[ranging_rows].sum(axis=1) == 0.0).all())
        self.assertTrue(
            (combined_alpha.loc[ranging_rows.values] == 0.0).all()
        )

    def test_fully_licensed_map_matches_plain_ensemble_within_each_regime(self):
        regime_alpha_map = {
            "trending": list(self.alpha_signals.keys()),
            "ranging": list(self.alpha_signals.keys()),
            "mixed": list(self.alpha_signals.keys())
        }
        conditional_weights, _, _ = (
            ensemble.build_regime_conditional_ensemble(
                self.alpha_signals, self.df, self.regime_labels,
                regime_alpha_map, method="equal"
            )
        )
        plain_weights = ensemble.equal_weights(
            ensemble.normalized_signal_matrix(self.alpha_signals)
        )

        pd.testing.assert_frame_equal(
            conditional_weights.reset_index(drop=True),
            plain_weights.reset_index(drop=True),
            check_dtype=False
        )


if __name__ == "__main__":
    unittest.main()
