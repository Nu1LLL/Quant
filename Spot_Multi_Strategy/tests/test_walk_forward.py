import unittest

import numpy as np
import pandas as pd

import alphas
import walk_forward
from alphas import base


def _make_ohlcv(rows, seed, trend=0.0002, scale=0.01, start_price=100.0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=trend, scale=scale, size=rows)
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


class WalkForwardFoldIsolationTests(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(rows=3000, seed=11)
        self.alpha_signals = alphas.build_single_asset_alphas(self.df)
        self.alpha = self.alpha_signals["A01_ts_momentum_24"]

    def test_mutating_a_later_fold_does_not_change_earlier_folds(self):
        result_before = walk_forward.evaluate_alpha_walk_forward(
            "TEST", self.alpha, self.df, fold_count=6
        )

        mutated_df = self.df.copy()
        # 只篡改最后10%的价格（属于最后一折），前面折不应该受影响
        cutoff = int(len(mutated_df) * 0.9)
        mutated_df.loc[cutoff:, ["open", "high", "low", "close"]] *= 3.0

        mutated_alphas = alphas.build_single_asset_alphas(mutated_df)
        mutated_alpha = mutated_alphas["A01_ts_momentum_24"]

        result_after = walk_forward.evaluate_alpha_walk_forward(
            "TEST", mutated_alpha, mutated_df, fold_count=6
        )

        early_folds_before = result_before.fold_table[
            result_before.fold_table["fold"] <= 3
        ].reset_index(drop=True)
        early_folds_after = result_after.fold_table[
            result_after.fold_table["fold"] <= 3
        ].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            early_folds_before[
                ["fold", "fold_ic", "fold_sharpe", "fold_pnl_sum",
                 "fold_turnover", "observations"]
            ],
            early_folds_after[
                ["fold", "fold_ic", "fold_sharpe", "fold_pnl_sum",
                 "fold_turnover", "observations"]
            ]
        )

    def test_fold_table_has_expected_fold_count(self):
        result = walk_forward.evaluate_alpha_walk_forward(
            "TEST", self.alpha, self.df, fold_count=6
        )
        self.assertEqual(len(result.fold_table), 6)
        self.assertListEqual(
            list(result.fold_table["fold"]), [1, 2, 3, 4, 5, 6]
        )


class AlphaGateLogicTests(unittest.TestCase):
    def _signal_from_series(self, name, normalized):
        return base.make_signal(
            name=name,
            raw=normalized,
            normalized=normalized,
            direction="trend",
            lookback=1
        )

    def test_gate_rejects_an_alpha_with_no_predictive_power(self):
        df = _make_ohlcv(rows=3000, seed=21, trend=0.0)
        rng = np.random.default_rng(99)
        noise_signal = pd.Series(rng.normal(size=len(df)))
        alpha_signal = self._signal_from_series(
            "noise_alpha", noise_signal.clip(-1, 1)
        )

        result = walk_forward.evaluate_alpha_walk_forward(
            "TEST", alpha_signal, df, fold_count=6
        )

        self.assertFalse(result.passed)

    def test_gate_accepts_an_alpha_with_genuine_edge(self):
        df = _make_ohlcv(rows=4000, seed=31, trend=0.0005, scale=0.006)
        # simulate_standalone_alpha用forward_return(horizon=1, delay=1)，
        # 也就是open[t+2]/open[t+1]-1，这里用完全相同的开盘价位移构造
        # 一个近乎完美的oracle信号，只用来测试gate的判定逻辑本身，
        # 不代表真实alpha——真实alpha永远不会用未来数据。
        future_direction = np.sign(
            df["open"].shift(-2) / df["open"].shift(-1) - 1
        ).fillna(0.0)
        oracle_signal = future_direction * 0.9
        alpha_signal = self._signal_from_series(
            "oracle_alpha", oracle_signal
        )

        result = walk_forward.evaluate_alpha_walk_forward(
            "TEST", alpha_signal, df, fold_count=6,
            min_total_observations=100, min_fold_observations=10
        )

        self.assertTrue(result.passed)
        self.assertGreater(result.summary["median_fold_ic"], 0)

    def test_single_fold_dominance_fails_the_gate(self):
        df = _make_ohlcv(rows=3000, seed=41)
        # 信号只在最后一折附近有效，其余全是0——集中在单折应该被拒绝
        zeros = pd.Series(0.0, index=df.index)
        fold_boundary = int(len(df) * 0.95)
        future_direction = np.sign(
            df["close"].shift(-2) / df["close"].shift(-1) - 1
        ).fillna(0.0)
        zeros.iloc[fold_boundary:] = (
            future_direction.iloc[fold_boundary:] * 0.9
        )
        alpha_signal = self._signal_from_series(
            "concentrated_alpha", zeros
        )

        result = walk_forward.evaluate_alpha_walk_forward(
            "TEST", alpha_signal, df, fold_count=6,
            min_total_observations=100, min_fold_observations=10
        )

        self.assertFalse(
            result.checks["单折PnL占比不超过50%"]
        )


if __name__ == "__main__":
    unittest.main()
