import unittest

import numpy as np
import pandas as pd

import alphas
import portfolio_backtest


def _make_ohlcv(rows, seed, trend=0.0004, scale=0.008, start_price=100.0):
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


class PortfolioBacktestPlumbingTests(unittest.TestCase):
    def setUp(self):
        self.frames = {
            "BTCUSDT": _make_ohlcv(2500, seed=1),
            "ETHUSDT": _make_ohlcv(2500, seed=2)
        }
        self.alpha_sets = {
            symbol: alphas.build_single_asset_alphas(df)
            for symbol, df in self.frames.items()
        }
        self.accepted = ["A02_ema_distance_50", "A08_trend_quality"]

    def test_dollar_equity_starts_at_initial_capital(self):
        result = portfolio_backtest.run_alpha_ensemble_portfolio(
            self.frames, self.alpha_sets, self.accepted, "equal",
            fee_rate=0.001, slippage_rate=0.0005, initial_capital=5000.0
        )
        equity = result["portfolio_equity"]["equity"]
        self.assertAlmostEqual(equity.iloc[0] * 5000.0, 5000.0, delta=50.0)

    def test_per_symbol_dollar_allocations_sum_to_portfolio(self):
        result = portfolio_backtest.run_alpha_ensemble_portfolio(
            self.frames, self.alpha_sets, self.accepted, "equal",
            fee_rate=0.001, slippage_rate=0.0005, initial_capital=5000.0
        )
        combined_dollar = result["portfolio_equity"]["equity"] * 5000.0
        per_symbol_sum = sum(
            simulation["dollar_equity"]
            for simulation in result["per_symbol"].values()
        )
        np.testing.assert_allclose(
            combined_dollar.values, per_symbol_sum.values, rtol=1e-6
        )

    def test_higher_cost_multiplier_never_improves_return(self):
        base = portfolio_backtest.run_alpha_ensemble_portfolio(
            self.frames, self.alpha_sets, self.accepted, "corr_penalized",
            fee_rate=0.001, slippage_rate=0.0005, initial_capital=5000.0
        )
        stressed = portfolio_backtest.run_alpha_ensemble_portfolio(
            self.frames, self.alpha_sets, self.accepted, "corr_penalized",
            fee_rate=0.003, slippage_rate=0.0015, initial_capital=5000.0
        )
        self.assertLessEqual(
            stressed["metrics"]["total_return"],
            base["metrics"]["total_return"] + 1e-9
        )
        self.assertGreaterEqual(
            stressed["metrics"]["total_fees"],
            base["metrics"]["total_fees"] - 1e-9
        )

    def test_buy_and_hold_matches_full_investment_no_rebalancing(self):
        result = portfolio_backtest.run_buy_and_hold_portfolio(
            self.frames, fee_rate=0.001, slippage_rate=0.0005,
            initial_capital=5000.0
        )
        self.assertEqual(int(result["portfolio_equity"]["rebalanced"].sum()), 1)

    def test_regime_filter_reduces_or_equals_raw_exposure(self):
        _, _, filtered_exposure, _ = portfolio_backtest.build_symbol_exposure(
            "BTCUSDT", self.frames["BTCUSDT"], self.alpha_sets,
            self.accepted, "equal", apply_regime_filter=True
        )
        _, _, raw_exposure, _ = portfolio_backtest.build_symbol_exposure(
            "BTCUSDT", self.frames["BTCUSDT"], self.alpha_sets,
            self.accepted, "equal", apply_regime_filter=False
        )
        comparison = (filtered_exposure <= raw_exposure + 1e-9).fillna(True)
        self.assertTrue(comparison.all())


if __name__ == "__main__":
    unittest.main()
