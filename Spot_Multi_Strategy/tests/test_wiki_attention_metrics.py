"""Synthetic-only checks; no Wikipedia values, market prices, or PnL read."""

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
import math
from statistics import stdev
import unittest
from unittest.mock import patch

import pandas as pd

from research.wiki_attention_metrics import (
    END_DAY_EXCLUSIVE, START_DAY, YEARS, MetricsDataError,
    _affected_windows, _annual_walk_forward, _btc_daily_closes,
    _monthly_stability, _regime_report, _returns_metrics,
    block_monte_carlo, evaluate,
)


@dataclass(frozen=True)
class Row:
    day: date
    equity_open: float
    equity_next_open: float
    return_net: float
    conservative_high_equity: float
    conservative_low_equity: float
    exposed: bool


@dataclass(frozen=True)
class Ledger:
    records: tuple[Row, ...]
    final_equity: float
    conservative_max_drawdown: float = 0.0
    conservative_max_drawdown_by_year: dict[int, float] = field(
        default_factory=lambda: {year: 0.0 for year in YEARS})
    fee_usdt: float = 0.0
    slippage_usdt: float = 0.0
    funding_usdt: float = 0.0
    turnover_usdt: float = 0.0
    trades: tuple[object, ...] = ()
    capacity_breaches: int = 0
    execution_breaches: int = 0
    margin_proxy_breaches: int = 0
    terminal_flat: bool = True


def synthetic_rows(return_for_day=lambda _: 0.0, *, exposed=False):
    rows = []
    equity = 10_000.0
    day = START_DAY
    while day < END_DAY_EXCLUSIVE:
        ending = equity * (1.0 + return_for_day(day))
        rows.append(Row(day, equity, ending, ending / equity - 1.0,
                        max(equity, ending), min(equity, ending), exposed))
        equity = ending
        day += timedelta(days=1)
    return tuple(rows)


def flat_ledger():
    rows = synthetic_rows()
    return Ledger(rows, rows[-1].equity_next_open)


def synthetic_btc_bars():
    index = pd.date_range("2020-12-01", "2026-01-01", freq="4h",
                          inclusive="left", tz="UTC", name="open_time")
    bars = pd.DataFrame({"close": 100.0}, index=index)
    terminal = pd.DataFrame({"close": [float("nan")]},
                            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")],
                                                   name="open_time"))
    return pd.concat((bars, terminal))


class WikiAttentionMetricsTests(unittest.TestCase):
    def test_cagr_sample_sharpe_daily_pf_and_undefined(self):
        values = [0.10, -0.05, 0.0, 0.02]
        metrics = _returns_metrics(values)
        multiple = math.prod(1.0 + item for item in values)
        self.assertAlmostEqual(metrics["cagr"], multiple ** (365.25 / 4) - 1)
        self.assertAlmostEqual(metrics["sharpe"],
                               math.sqrt(365.25) * sum(values) / 4 / stdev(values))
        self.assertAlmostEqual(metrics["profit_factor_daily"], 2.4)
        self.assertAlmostEqual(metrics["max_drawdown_daily"], -0.05)
        flat = _returns_metrics([0.0] * 4)
        self.assertIsNone(flat["sharpe"])
        self.assertIsNone(flat["profit_factor_daily"])
        self.assertEqual(flat["sharpe_status"], "zero_sample_variance")
        self.assertEqual(flat["profit_factor_status"], "no_negative_days")

    def test_annual_fold_requires_joint_criteria_and_uses_4h_stress(self):
        def returns(day):
            if day.day == 1 and day.month == 1:
                return 0.02
            if day.day == 2 and day.month == 1:
                return -0.001
            return 0.0

        rows = synthetic_rows(returns)
        annual_dd = {year: -0.05 for year in YEARS}
        annual_dd[2024] = -0.21
        folds = _annual_walk_forward(rows, annual_dd)
        self.assertEqual([item["year"] for item in folds], list(YEARS))
        self.assertEqual(sum(item["passed"] for item in folds), 4)
        self.assertFalse(next(item for item in folds if item["year"] == 2024)["passed"])
        self.assertEqual(next(item for item in folds if item["year"] == 2024)[
            "conservative_max_drawdown"], -0.21)

    def test_months_and_wikimedia_windows_do_not_delete_days(self):
        rows = synthetic_rows()
        months = _monthly_stability(rows)
        self.assertEqual(months["complete_months"], 60)
        self.assertEqual(months["positive_months"], 0)
        self.assertFalse(months["passed"])
        windows = _affected_windows(rows)
        self.assertEqual(windows["wikimedia_2021_2022_traffic_loss"]["start"], "2021-06-04")
        self.assertEqual(windows["wikimedia_2025_source_change"]["end_inclusive"],
                         "2025-12-31")
        self.assertFalse(windows["wikimedia_2025_source_change"]["excluded_from_analysis"])

    def test_btc_close_grid_requires_all_4h_bars_but_excludes_terminal_open(self):
        bars = synthetic_btc_bars()
        closes = _btc_daily_closes(bars)
        self.assertEqual(closes[date(2020, 12, 1)], 100.0)
        self.assertEqual(closes[date(2025, 12, 31)], 100.0)
        with self.assertRaisesRegex(MetricsDataError, "4h gap"):
            _btc_daily_closes(bars.drop(bars.index[10]))
        with self.assertRaisesRegex(MetricsDataError, "invalid value"):
            broken = bars.copy()
            broken.loc[pd.Timestamp("2022-01-01", tz="UTC"), "close"] = float("nan")
            _btc_daily_closes(broken)

    def test_regime_uses_previous_30_complete_utc_closes(self):
        rows = [Row(date(2021, 1, day), 100.0, 100.0, 0.0,
                    100.0, 100.0, day == 1) for day in range(1, 4)]
        closes = {
            date(2020, 12, 1): 100.0, date(2020, 12, 2): 100.0,
            date(2020, 12, 3): 100.0, date(2020, 12, 31): 111.0,
            date(2021, 1, 1): 89.0, date(2021, 1, 2): 100.0,
        }
        result = _regime_report(rows, closes)
        self.assertEqual([result[name]["days"] for name in ("bull", "bear", "sideways")],
                         [1, 1, 1])
        self.assertEqual(result["bull"]["exposure_days"], 1)

    def test_circular_block_monte_carlo_is_reproducible_and_undefined_fails(self):
        values = [0.02, -0.01, 0.0, 0.015, -0.005] * 8
        first = block_monte_carlo(values, paths=73, block_days=7, seed=20261102)
        second = block_monte_carlo(values, paths=73, block_days=7, seed=20261102)
        self.assertEqual(first, second)
        self.assertEqual(first["joint_numeric_gate_probability"],
                         first["joint_numeric_gate_paths"] / 73)
        self.assertEqual(first["drawdown_basis"], "daily_close_only_not_main_4h_stress")
        flat = block_monte_carlo([0.0] * 30, paths=20)
        self.assertIsNone(flat["p05_sharpe"])
        self.assertIsNone(flat["p05_profit_factor_daily"])
        self.assertFalse(flat["passed"])
        no_losses = block_monte_carlo([0.001] * 30, paths=20)
        self.assertFalse(no_losses["passed"])

    def test_full_evaluation_oos_lock_fixed_period_and_missing_benchmark(self):
        ledger = flat_ledger()
        bars = synthetic_btc_bars()
        with patch("research.wiki_attention_metrics.block_monte_carlo",
                   return_value={"paths": 10_000, "block_days": 21, "seed": 20261102,
                                 "passed": False}):
            report = evaluate(ledger, ledger, bars)
            without_benchmark = evaluate(ledger, None, bars)
        self.assertEqual(report["strategy"]["days"], 1826)
        self.assertEqual(report["monthly_stability"]["complete_months"], 60)
        self.assertEqual(report["historical_time_holdout_2024_2025"]["days"], 731)
        self.assertEqual(report["regimes"]["sideways"]["days"], 1826)
        self.assertFalse(report["independent_oos"])
        self.assertFalse(report["full_gate"])
        self.assertTrue(report["benchmark_path_complete"])
        self.assertIsNone(without_benchmark["benchmark"])
        self.assertFalse(without_benchmark["benchmark_path_complete"])
        self.assertEqual(report["strategy_execution"]["trades"], 0)

    def test_fail_closed_on_grid_nan_and_execution_breaches(self):
        ledger = flat_ledger()
        bars = synthetic_btc_bars()
        with self.assertRaisesRegex(MetricsDataError, "daily grid"):
            evaluate(replace(ledger, records=ledger.records[:-1]), ledger, bars)
        broken = list(ledger.records)
        broken[4] = replace(broken[4], return_net=float("nan"))
        with self.assertRaisesRegex(MetricsDataError, "invalid strategy daily return"):
            evaluate(replace(ledger, records=tuple(broken)), ledger, bars)
        with patch("research.wiki_attention_metrics.block_monte_carlo",
                   return_value={"passed": False}):
            report = evaluate(replace(ledger, execution_breaches=1), ledger, bars)
            one_trade = evaluate(replace(ledger, trades=(object(),)), ledger, bars)
            nonflat = evaluate(replace(ledger, terminal_flat=False), ledger, bars)
        self.assertFalse(report["gate_checks"]["minimum_order_execution"])
        self.assertFalse(report["numerical_gate"])
        self.assertEqual(one_trade["strategy_execution"]["trades"], 1)
        self.assertFalse(nonflat["gate_checks"]["terminal_flat"])


if __name__ == "__main__":
    unittest.main()
