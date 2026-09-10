import unittest

import numpy as np
import pandas as pd

import portfolio_metrics


def _simulation_df(rows, daily_return, seed=0, with_extra_columns=True):
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.001, size=rows)
    net_pnl = np.full(rows, daily_return) + noise
    open_time = pd.date_range("2021-01-01", periods=rows, freq="4h", tz="UTC")

    data = {"open_time": open_time, "net_pnl": net_pnl}
    if with_extra_columns:
        data["position"] = np.full(rows, 0.5)
        data["turnover"] = np.where(np.arange(rows) % 20 == 0, 0.1, 0.0)
        data["cost"] = data["turnover"] * 0.003
        data["rebalanced"] = data["turnover"] > 0

    return pd.DataFrame(data)


class PortfolioMetricsTests(unittest.TestCase):
    def test_positive_drift_produces_positive_cagr_and_sharpe(self):
        df = _simulation_df(2000, daily_return=0.0005, seed=1)
        metrics = portfolio_metrics.calculate_extended_metrics(df)

        self.assertGreater(metrics["cagr"], 0)
        self.assertGreater(metrics["sharpe_ratio"], 0)
        self.assertGreater(metrics["sortino_ratio"], 0)

    def test_max_drawdown_is_non_positive(self):
        df = _simulation_df(2000, daily_return=0.0002, seed=2)
        metrics = portfolio_metrics.calculate_extended_metrics(df)
        self.assertLessEqual(metrics["max_drawdown"], 0.0)
        self.assertLessEqual(metrics["average_drawdown"], 0.0)

    def test_negative_drift_produces_negative_calmar_input(self):
        df = _simulation_df(2000, daily_return=-0.001, seed=3)
        metrics = portfolio_metrics.calculate_extended_metrics(df)
        self.assertLess(metrics["cagr"], 0)
        self.assertLess(metrics["max_drawdown"], 0)

    def test_monthly_hit_rate_is_bounded(self):
        df = _simulation_df(3000, daily_return=0.0003, seed=4)
        metrics = portfolio_metrics.calculate_extended_metrics(df)
        self.assertGreaterEqual(metrics["monthly_hit_rate"], 0.0)
        self.assertLessEqual(metrics["monthly_hit_rate"], 1.0)

    def test_missing_optional_columns_do_not_crash(self):
        df = _simulation_df(500, daily_return=0.0001, seed=5, with_extra_columns=False)
        metrics = portfolio_metrics.calculate_extended_metrics(df)
        self.assertTrue(np.isnan(metrics["turnover_per_bar"]))
        self.assertTrue(np.isnan(metrics["total_fees"]))

    def test_total_fees_scale_with_initial_capital(self):
        df = _simulation_df(500, daily_return=0.0001, seed=6)
        small = portfolio_metrics.calculate_extended_metrics(df, initial_capital=1.0)
        large = portfolio_metrics.calculate_extended_metrics(df, initial_capital=1000.0)
        self.assertAlmostEqual(large["total_fees"], small["total_fees"] * 1000, places=6)


if __name__ == "__main__":
    unittest.main()
