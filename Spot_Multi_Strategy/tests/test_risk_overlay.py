import unittest

import numpy as np
import pandas as pd

import risk_overlay


def _make_ohlcv(rows, seed, scale=0.01, start_price=100.0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0, scale=scale, size=rows)
    close = start_price * np.cumprod(1 + returns)

    open_ = np.empty(rows)
    open_[0] = start_price
    open_[1:] = close[:-1]

    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
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


class VolatilityTargetingTests(unittest.TestCase):
    def test_scalar_is_one_when_vol_targeting_disabled(self):
        df = _make_ohlcv(200, seed=1)
        scalar = risk_overlay.realized_volatility_scalar(
            df["close"], None, 20, 2000
        )
        self.assertTrue((scalar == 1.0).all())

    def test_scalar_never_exceeds_one(self):
        df = _make_ohlcv(500, seed=2, scale=0.001)
        scalar = risk_overlay.realized_volatility_scalar(
            df["close"], vol_target_annualized=0.3,
            vol_window=20, periods_per_year=2190
        )
        self.assertTrue((scalar <= 1.0 + 1e-9).all())

    def test_scalar_drops_when_realized_vol_spikes(self):
        rows = 400
        close = np.concatenate([
            100 * np.cumprod(1 + np.random.default_rng(3).normal(0, 0.001, rows // 2)),
        ])
        # 前半段低波动，后半段用固定大跳动制造高波动
        high_vol_returns = np.tile([0.05, -0.05], rows // 4)
        close_high = close[-1] * np.cumprod(1 + high_vol_returns)
        full_close = np.concatenate([close, close_high])

        df = pd.DataFrame({
            "open_time": pd.date_range(
                "2020-01-01", periods=len(full_close), freq="4h", tz="UTC"
            ),
            "open": full_close,
            "high": full_close * 1.001,
            "low": full_close * 0.999,
            "close": full_close,
            "volume": 100.0
        })

        scalar = risk_overlay.realized_volatility_scalar(
            df["close"], vol_target_annualized=0.3,
            vol_window=20, periods_per_year=2190
        )

        low_vol_scalar = scalar.iloc[100:150].mean()
        high_vol_scalar = scalar.iloc[-50:].mean()
        self.assertLess(high_vol_scalar, low_vol_scalar)


class RiskOverlaySimulationTests(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(2000, seed=10, scale=0.01)

    def test_position_never_exceeds_exposure_cap(self):
        raw_exposure = pd.Series(1.0, index=self.df.index)
        config = risk_overlay.RiskConfig(exposure_cap=0.6, no_trade_band=0.0)
        result = risk_overlay.apply_risk_overlay(self.df, raw_exposure, config)
        self.assertTrue((result["position"] <= 0.6 + 1e-9).all())

    def test_no_trade_band_suppresses_small_changes(self):
        rng = np.random.default_rng(4)
        raw_exposure = pd.Series(
            0.5 + rng.normal(scale=0.01, size=len(self.df))
        ).clip(0, 1)
        config = risk_overlay.RiskConfig(no_trade_band=0.2, exposure_cap=1.0)
        result = risk_overlay.apply_risk_overlay(self.df, raw_exposure, config)

        distinct_positions = result["position"].nunique()
        self.assertLess(distinct_positions, 10)

    def test_tight_no_trade_band_tracks_target_closely(self):
        rng = np.random.default_rng(4)
        raw_exposure = pd.Series(
            0.5 + rng.normal(scale=0.2, size=len(self.df))
        ).clip(0, 1)
        loose_config = risk_overlay.RiskConfig(no_trade_band=0.5)
        tight_config = risk_overlay.RiskConfig(no_trade_band=0.01)

        loose_result = risk_overlay.apply_risk_overlay(
            self.df, raw_exposure, loose_config
        )
        tight_result = risk_overlay.apply_risk_overlay(
            self.df, raw_exposure, tight_config
        )

        self.assertGreater(
            tight_result["position"].nunique(),
            loose_result["position"].nunique()
        )

    def test_drawdown_control_reduces_exposure_after_losses(self):
        rows = 600
        # 构造一段持续下跌的行情，让模拟组合真实产生回撤
        crash_returns = np.full(rows, -0.01)
        close = 100 * np.cumprod(1 + crash_returns)
        df = pd.DataFrame({
            "open_time": pd.date_range(
                "2020-01-01", periods=rows, freq="4h", tz="UTC"
            ),
            "open": np.concatenate([[100.0], close[:-1]]),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 100.0
        })

        raw_exposure = pd.Series(1.0, index=df.index)
        config = risk_overlay.RiskConfig(
            drawdown_soft_threshold=0.05,
            drawdown_hard_threshold=0.15,
            drawdown_min_scalar=0.2,
            no_trade_band=0.0
        )
        result = risk_overlay.apply_risk_overlay(df, raw_exposure, config)

        early_scalar = result["drawdown_scalar"].iloc[:20].mean()
        late_scalar = result["drawdown_scalar"].iloc[-20:].mean()
        self.assertLess(late_scalar, early_scalar)
        self.assertGreaterEqual(late_scalar, config.drawdown_min_scalar - 1e-9)

    def test_zero_exposure_produces_zero_pnl(self):
        raw_exposure = pd.Series(0.0, index=self.df.index)
        result = risk_overlay.apply_risk_overlay(self.df, raw_exposure)
        self.assertTrue((result["net_pnl"].fillna(0.0) == 0.0).all())


class StopLossTests(unittest.TestCase):
    def _crash_after_entry_df(self, flat_rows=60, crash_rows=40, crash_per_bar=-0.02):
        flat_close = np.full(flat_rows, 100.0)
        crash_close = 100.0 * np.cumprod(
            1 + np.full(crash_rows, crash_per_bar)
        )
        close = np.concatenate([flat_close, crash_close])
        rows = len(close)

        return pd.DataFrame({
            "open_time": pd.date_range(
                "2020-01-01", periods=rows, freq="4h", tz="UTC"
            ),
            "open": np.concatenate([[100.0], close[:-1]]),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 100.0
        })

    def test_stop_loss_forces_exit_and_caps_drawdown(self):
        df = self._crash_after_entry_df()
        raw_exposure = pd.Series(1.0, index=df.index)

        no_stop = risk_overlay.apply_risk_overlay(
            df, raw_exposure,
            risk_overlay.RiskConfig(no_trade_band=0.0, stop_loss_atr_multiple=None)
        )
        with_stop = risk_overlay.apply_risk_overlay(
            df, raw_exposure,
            risk_overlay.RiskConfig(
                no_trade_band=0.0, stop_loss_atr_multiple=1.5,
                stop_loss_atr_window=14
            )
        )

        self.assertTrue(with_stop["stopped_out"].any())
        # raw_exposure恒为1，止损归零后下一根就会因为信号依然乐观
        # 重新建仓，所以这里不检查"从此保持空仓"，只检查止损版本的
        # 最大回撤明显小于不设止损的版本。
        no_stop_drawdown = (
            no_stop["equity"] / no_stop["equity"].cummax() - 1
        ).min()
        with_stop_drawdown = (
            with_stop["equity"] / with_stop["equity"].cummax() - 1
        ).min()

        self.assertLess(with_stop_drawdown, 0)
        self.assertGreater(with_stop_drawdown, no_stop_drawdown)

    def test_no_stop_loss_when_multiplier_is_none(self):
        df = self._crash_after_entry_df()
        raw_exposure = pd.Series(1.0, index=df.index)
        result = risk_overlay.apply_risk_overlay(
            df, raw_exposure,
            risk_overlay.RiskConfig(stop_loss_atr_multiple=None)
        )
        self.assertFalse(result["stopped_out"].any())

    def test_stop_loss_overrides_no_trade_band(self):
        df = self._crash_after_entry_df(crash_per_bar=-0.05)
        raw_exposure = pd.Series(1.0, index=df.index)
        result = risk_overlay.apply_risk_overlay(
            df, raw_exposure,
            risk_overlay.RiskConfig(
                no_trade_band=0.9,  # 大到几乎永远不会因为普通换手触发
                stop_loss_atr_multiple=1.0
            )
        )
        self.assertTrue(result["stopped_out"].any())
        stop_index = result.index[result["stopped_out"]][0]
        self.assertEqual(result.loc[stop_index, "position"], 0.0)

    def test_stop_loss_is_causal(self):
        df = self._crash_after_entry_df()
        raw_exposure = pd.Series(1.0, index=df.index)
        config = risk_overlay.RiskConfig(
            no_trade_band=0.0, stop_loss_atr_multiple=1.5
        )

        before = risk_overlay.apply_risk_overlay(df, raw_exposure, config)

        mutated_df = df.copy()
        cutoff = int(len(df) * 0.9)
        mutated_df.loc[cutoff:, ["open", "high", "low", "close"]] *= 5.0

        after = risk_overlay.apply_risk_overlay(mutated_df, raw_exposure, config)

        safe_boundary = cutoff - 30
        pd.testing.assert_series_equal(
            before["position"].iloc[:safe_boundary],
            after["position"].iloc[:safe_boundary]
        )
        pd.testing.assert_series_equal(
            before["stopped_out"].iloc[:safe_boundary],
            after["stopped_out"].iloc[:safe_boundary]
        )


class AlphaConcentrationCapTests(unittest.TestCase):
    def test_no_single_alpha_exceeds_cap(self):
        weights = pd.DataFrame({
            "a": [0.8, 0.9],
            "b": [0.1, 0.05],
            "c": [0.1, 0.05]
        })
        capped = risk_overlay.cap_alpha_concentration(weights, max_weight=0.5)
        self.assertTrue((capped["a"] <= 0.5 + 1e-6).all())

    def test_no_cap_when_max_weight_is_one(self):
        weights = pd.DataFrame({"a": [0.8], "b": [0.2]})
        capped = risk_overlay.cap_alpha_concentration(weights, max_weight=1.0)
        pd.testing.assert_frame_equal(capped, weights)


if __name__ == "__main__":
    unittest.main()
