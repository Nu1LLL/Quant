import unittest

import numpy as np
import pandas as pd

import alphas
import market_neutral as mn


def _make_ohlcv(rows, seed, start_price=100.0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=0.01, size=rows)
    close = start_price * np.cumprod(1 + returns)

    open_ = np.empty(rows)
    open_[0] = start_price
    open_[1:] = close[:-1]

    open_time = pd.date_range(start, periods=rows, freq="4h", tz="UTC")

    return pd.DataFrame({
        "open_time": open_time,
        "open": open_,
        "high": np.maximum(open_, close) * 1.002,
        "low": np.minimum(open_, close) * 0.998,
        "close": close,
        "volume": rng.uniform(100, 1000, rows)
    })


class CrossSectionalConstructionTests(unittest.TestCase):
    def setUp(self):
        self.frames = {
            "AAAUSDT": _make_ohlcv(1000, seed=1, start_price=100.0),
            "BBBUSDT": _make_ohlcv(1000, seed=2, start_price=50.0),
            "CCCUSDT": _make_ohlcv(1000, seed=3, start_price=10.0),
        }
        self.alpha_sets = {
            symbol: alphas.build_single_asset_alphas(df)
            for symbol, df in self.frames.items()
        }

    def test_cross_sectional_demean_sums_to_zero_each_row(self):
        signal = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A02_ema_distance_50"
        )
        row_sums = signal.dropna().sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 0.0, atol=1e-9))

    def test_gross_cap_never_exceeds_cap_and_never_scales_up(self):
        signal = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A02_ema_distance_50"
        )
        capped = mn.scale_to_gross_cap(signal, gross_cap=1.0)
        gross = capped.abs().sum(axis=1)
        self.assertTrue((gross <= 1.0 + 1e-9).all())

        # 缩放只能缩小，不能放大：任何一行的绝对值都不应该比裁剪前更大
        # （NaN比较本身恒为False，用fillna(True)避免NaN单元格误报）
        raw_clipped = signal.clip(-1, 1)
        comparison = (capped.abs() <= raw_clipped.abs() + 1e-9).where(
            raw_clipped.notna(), True
        )
        self.assertTrue(comparison.all().all())

    def test_no_trade_band_reduces_turnover_events(self):
        signal = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A02_ema_distance_50"
        )
        capped = mn.scale_to_gross_cap(signal, gross_cap=1.0)

        _, tight_turnover = mn.apply_no_trade_band(capped, no_trade_band=0.0)
        _, loose_turnover = mn.apply_no_trade_band(capped, no_trade_band=0.3)

        self.assertGreater(
            (tight_turnover > 0).sum().sum(),
            (loose_turnover > 0).sum().sum()
        )

    def test_late_starting_asset_excluded_before_its_own_history(self):
        late_frames = dict(self.frames)
        late_frames["DDDUSDT"] = _make_ohlcv(
            1000, seed=4, start_price=1.0
        ).iloc[200:].reset_index(drop=True)
        late_alpha_sets = dict(self.alpha_sets)
        late_alpha_sets["DDDUSDT"] = alphas.build_single_asset_alphas(
            late_frames["DDDUSDT"]
        )

        signal = mn.build_cross_sectional_signal(
            late_frames, late_alpha_sets, "A02_ema_distance_50"
        )
        early_rows = signal.iloc[:50]
        self.assertTrue(early_rows["DDDUSDT"].isna().all())
        # 其它资产不应该因为DDD还没数据就被污染成全NaN
        self.assertFalse(early_rows["AAAUSDT"].isna().all())


class NoLookaheadTests(unittest.TestCase):
    def setUp(self):
        self.frames = {
            "AAAUSDT": _make_ohlcv(800, seed=5, start_price=100.0),
            "BBBUSDT": _make_ohlcv(800, seed=6, start_price=50.0),
            "CCCUSDT": _make_ohlcv(800, seed=7, start_price=10.0),
        }
        self.alpha_sets = {
            symbol: alphas.build_single_asset_alphas(df)
            for symbol, df in self.frames.items()
        }

    def test_signal_truncation_by_calendar_time_is_causal(self):
        full = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A08_trend_quality"
        )
        cutoff = self.frames["AAAUSDT"]["open_time"].iloc[-100]
        truncated_frames = {
            symbol: frame[frame["open_time"] < cutoff].reset_index(drop=True)
            for symbol, frame in self.frames.items()
        }
        truncated_alpha_sets = {
            symbol: alphas.build_single_asset_alphas(df)
            for symbol, df in truncated_frames.items()
        }
        truncated = mn.build_cross_sectional_signal(
            truncated_frames, truncated_alpha_sets, "A08_trend_quality"
        )

        overlap = truncated.index
        pd.testing.assert_frame_equal(
            full.loc[overlap], truncated, check_dtype=False
        )

    def test_apply_no_trade_band_is_causal(self):
        signal = mn.build_cross_sectional_signal(
            self.frames, self.alpha_sets, "A08_trend_quality"
        )
        capped = mn.scale_to_gross_cap(signal)

        full_pos, _ = mn.apply_no_trade_band(capped, no_trade_band=0.05)

        cutoff_position = len(capped) - 100
        truncated_capped = capped.iloc[:cutoff_position]
        truncated_pos, _ = mn.apply_no_trade_band(
            truncated_capped, no_trade_band=0.05
        )

        pd.testing.assert_frame_equal(
            full_pos.iloc[:cutoff_position], truncated_pos, check_dtype=False
        )


class FundingCostSignTests(unittest.TestCase):
    def test_long_pays_when_funding_positive_short_receives(self):
        rows = 200
        frame = _make_ohlcv(rows, seed=8)
        funding_time = pd.date_range(
            "2020-01-01", periods=rows // 2, freq="8h", tz="UTC"
        )
        funding_df = pd.DataFrame({
            "funding_time": funding_time,
            "funding_rate": np.full(len(funding_time), 0.001)
        })

        frames = {"AAAUSDT": frame}
        funding_frames = {"AAAUSDT": funding_df}

        long_weights = pd.DataFrame(
            {"AAAUSDT": np.full(rows, 0.5)},
            index=pd.to_datetime(frame["open_time"], utc=True)
        )
        short_weights = pd.DataFrame(
            {"AAAUSDT": np.full(rows, -0.5)},
            index=pd.to_datetime(frame["open_time"], utc=True)
        )

        long_result = mn.simulate_market_neutral_portfolio(
            frames, funding_frames, long_weights, no_trade_band=0.0
        )
        short_result = mn.simulate_market_neutral_portfolio(
            frames, funding_frames, short_weights, no_trade_band=0.0
        )

        long_funding = long_result["AAAUSDT"]["funding_pnl"].iloc[1:-3].sum()
        short_funding = short_result["AAAUSDT"]["funding_pnl"].iloc[1:-3].sum()

        self.assertLess(long_funding, 0)
        self.assertGreater(short_funding, 0)
        self.assertAlmostEqual(long_funding, -short_funding, places=6)


class GateSmokeTests(unittest.TestCase):
    def test_evaluate_cross_sectional_alpha_runs_and_reports_checks(self):
        frames = {
            "AAAUSDT": _make_ohlcv(3500, seed=9, start_price=100.0),
            "BBBUSDT": _make_ohlcv(3500, seed=10, start_price=50.0),
            "CCCUSDT": _make_ohlcv(3500, seed=11, start_price=10.0),
        }
        alpha_sets = {
            symbol: alphas.build_single_asset_alphas(df)
            for symbol, df in frames.items()
        }
        result = mn.evaluate_cross_sectional_alpha(
            "A02_ema_distance_50", frames, alpha_sets,
            min_total_observations=100, min_fold_observations=10
        )
        self.assertIsInstance(result.passed, (bool, np.bool_))
        self.assertEqual(len(result.checks), 7)
        self.assertEqual(len(result.fold_table), 6)


if __name__ == "__main__":
    unittest.main()
