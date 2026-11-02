"""Synthetic checks for the frozen CME BTC positioning diagnostic formulas."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
from statistics import stdev
import unittest

from research.cme_btc_metrics import (
    END_DAY_EXCLUSIVE,
    START_DAY,
    MetricsDataError,
    _annual_walk_forward,
    _monthly_stability,
    _regimes,
    _returns_metrics,
    _stressed_drawdown,
    block_monte_carlo,
    evaluate_diagnostics,
)


UTC = timezone.utc


@dataclass(frozen=True)
class Row:
    day: date
    equity_open: float
    equity_next_open: float
    return_net: float
    exposed: bool
    conservative_drawdown: float
    conservative_high_equity: float
    conservative_low_equity: float
    regime_reference_close: float | None = None


@dataclass(frozen=True)
class Ledger:
    records: tuple[Row, ...]
    final_equity: float
    conservative_max_drawdown: float
    capacity_pass: bool = True
    execution_pass: bool = True


@dataclass(frozen=True)
class Bar:
    open_time: datetime
    close: float


def synthetic_rows(return_for_day, *, exposed=False):
    rows = []
    equity = 10_000.0
    day = START_DAY
    while day < END_DAY_EXCLUSIVE:
        ret = return_for_day(day)
        next_equity = equity * (1.0 + ret)
        rows.append(Row(day, equity, next_equity, next_equity / equity - 1.0,
                        exposed, 0.0, equity, equity))
        equity = next_equity
        day += timedelta(days=1)
    return tuple(rows)


class CmeBtcMetricsTests(unittest.TestCase):
    def test_cagr_sample_sharpe_daily_pf_and_undefined_cases(self):
        values = [0.10, -0.05, 0.0, 0.02]
        actual = _returns_metrics(values)
        multiple = math.prod(1 + value for value in values)
        self.assertAlmostEqual(actual["cagr"], multiple ** (365.25 / 4) - 1)
        self.assertAlmostEqual(actual["sharpe"],
                               math.sqrt(365.25) * sum(values) / 4 / stdev(values))
        self.assertAlmostEqual(actual["profit_factor_daily"], 2.4)
        self.assertAlmostEqual(actual["max_drawdown_daily"], -0.05)
        self.assertEqual(actual["zero_days"], 1)
        flat = _returns_metrics([0.0, 0.0, 0.0])
        self.assertIsNone(flat["sharpe"])
        self.assertIsNone(flat["profit_factor_daily"])
        self.assertEqual(flat["sharpe_status"], "zero_sample_variance")
        self.assertEqual(flat["profit_factor_status"], "no_negative_days")

    def test_adverse_high_before_low_and_fold_peak_reset(self):
        first = Row(date(2019, 1, 1), 100.0, 110.0, 0.10, True,
                    -0.25, 120.0, 90.0)
        second = Row(date(2020, 1, 1), 110.0, 109.0, 109 / 110 - 1, True,
                     -0.25, 115.0, 105.0)
        self.assertAlmostEqual(_stressed_drawdown((first,)), -0.25)
        self.assertAlmostEqual(_stressed_drawdown((second,)), 105 / 115 - 1)

    def test_annual_six_of_seven_requires_joint_fold_conditions(self):
        def yearly_returns(day):
            if day.month == 1 and day.day == 1:
                return 0.001 if day.year == 2022 else 0.02
            if day.month == 1 and day.day == 2:
                return -0.02 if day.year == 2022 else -0.001
            return 0.0

        folds = _annual_walk_forward(synthetic_rows(yearly_returns))
        self.assertEqual([fold["year"] for fold in folds], list(range(2019, 2026)))
        self.assertEqual(sum(fold["passed"] for fold in folds), 6)
        self.assertFalse(next(fold for fold in folds if fold["year"] == 2022)["passed"])

    def test_monthly_positive_ratio_worst_and_loss_streak(self):
        returns = {1: 0.10, 2: -0.05, 3: -0.02, 4: 0.0, 5: 0.03}
        rows = []
        equity = 100.0
        for month, ret in returns.items():
            day = date(2020, month, 1)
            ending = equity * (1 + ret)
            rows.append(Row(day, equity, ending, ret, False, 0.0,
                            max(equity, ending), min(equity, ending)))
            equity = ending
        result = _monthly_stability(rows)
        self.assertEqual(result["positive_months"], 2)
        self.assertAlmostEqual(result["positive_month_ratio"], 0.4)
        self.assertEqual(result["worst_month"], "2020-02")
        self.assertAlmostEqual(result["worst_month_return"], -0.05)
        self.assertEqual(result["longest_consecutive_losing_months"], 2)
        self.assertFalse(result["passed"])

    def test_regimes_use_only_previous_thirty_complete_closes(self):
        closes = {
            date(2018, 12, 1): 100.0,
            date(2018, 12, 2): 100.0,
            date(2018, 12, 3): 100.0,
            date(2018, 12, 4): 100.0,
            date(2018, 12, 31): 111.0,
            date(2019, 1, 1): 89.0,
            date(2019, 1, 2): 110.0,
            date(2019, 1, 3): 90.0,
        }
        rows = [Row(date(2019, 1, day), 100.0, 100.0, 0.0, day == 1,
                    0.0, 100.0, 100.0) for day in range(1, 5)]
        result = _regimes(rows, closes)
        self.assertEqual([result[name]["days"] for name in ("bull", "bear", "sideways")],
                         [1, 1, 2])
        self.assertEqual(result["bull"]["exposure_days"], 1)

    def test_monte_carlo_fixed_seed_circular_blocks_and_joint_gate(self):
        values = [0.02, -0.01, 0.0, 0.015, -0.005] * 8
        first = block_monte_carlo(values, paths=73, block_days=7, seed=20261102)
        second = block_monte_carlo(values, paths=73, block_days=7, seed=20261102)
        self.assertEqual(first, second)
        self.assertEqual(first["paths"], 73)
        self.assertEqual(first["block_days"], 7)
        self.assertEqual(first["drawdown_basis"], "daily_close_only")
        self.assertAlmostEqual(first["joint_numeric_gate_probability"],
                               first["joint_numeric_gate_paths"] / 73)
        flat = block_monte_carlo([0.0] * 30, paths=20, block_days=21)
        self.assertEqual(flat["joint_numeric_gate_probability"], 0.0)
        self.assertEqual(flat["p05_cagr"], 0.0)
        self.assertIsNone(flat["p05_sharpe"])
        self.assertIsNone(flat["p05_profit_factor_daily"])
        self.assertFalse(flat["passed"])
        constant_positive = block_monte_carlo([0.0001] * 30, paths=20, block_days=21)
        self.assertIsNone(constant_positive["p05_sharpe"])
        self.assertIsNone(constant_positive["p05_profit_factor_daily"])
        self.assertEqual(constant_positive["joint_numeric_gate_probability"], 0.0)

    def test_full_evaluation_has_fixed_calendar_benchmark_and_oos_lock(self):
        strategy_rows = synthetic_rows(lambda _: 0.0)
        benchmark_rows = synthetic_rows(lambda _: 0.0001, exposed=True)
        bars = []
        day = START_DAY - timedelta(days=31)
        while day < END_DAY_EXCLUSIVE:
            bars.append(Bar(datetime.combine(day, time.min, UTC), 100.0))
            day += timedelta(days=1)
        report = evaluate_diagnostics(
            Ledger(strategy_rows, strategy_rows[-1].equity_next_open, 0.0),
            tuple(bars),
            Ledger(benchmark_rows, benchmark_rows[-1].equity_next_open, 0.0),
        )
        self.assertEqual(report["strategy"]["days"], (END_DAY_EXCLUSIVE - START_DAY).days)
        self.assertEqual(report["monthly_stability"]["complete_months"], 84)
        self.assertEqual(report["annual_walk_forward_passes"], 0)
        self.assertEqual(len(report["annual_walk_forward"]), 7)
        self.assertEqual(report["historical_time_holdout_2022_2025"]["days"], 1461)
        self.assertEqual(report["regimes"]["sideways"]["days"], report["strategy"]["days"])
        self.assertGreater(report["benchmark"]["cagr"], 0)
        self.assertEqual(report["monte_carlo"]["paths"], 10_000)
        self.assertEqual(report["monte_carlo"]["block_days"], 21)
        self.assertEqual(report["monte_carlo"]["seed"], 20261102)
        self.assertFalse(report["numerical_gate"])
        self.assertTrue(report["gate_checks"]["minimum_order_execution"])
        self.assertFalse(report["independent_oos"])
        self.assertFalse(report["full_gate"])

    def test_minimum_order_execution_failure_fails_gate(self):
        rows = synthetic_rows(lambda _: 0.0)
        bars = []
        day = START_DAY - timedelta(days=31)
        while day < END_DAY_EXCLUSIVE:
            bars.append(Bar(datetime.combine(day, time.min, UTC), 100.0))
            day += timedelta(days=1)
        report = evaluate_diagnostics(
            Ledger(rows, rows[-1].equity_next_open, 0.0, execution_pass=False),
            tuple(bars), Ledger(rows, rows[-1].equity_next_open, 0.0),
        )
        self.assertFalse(report["gate_checks"]["minimum_order_execution"])
        self.assertFalse(report["numerical_gate"])

    def test_rejects_incomplete_or_inconsistent_ledger_before_evaluation(self):
        rows = synthetic_rows(lambda _: 0.0)
        with self.assertRaisesRegex(MetricsDataError, "daily grid"):
            evaluate_diagnostics(Ledger(rows[:-1], 10_000.0, 0.0), (),
                                 Ledger(rows, 10_000.0, 0.0))
        broken = list(rows)
        broken[3] = Row(broken[3].day, 10_000.0, 10_000.0, 0.1, False,
                        0.0, 10_000.0, 10_000.0)
        with self.assertRaisesRegex(MetricsDataError, "inconsistent"):
            evaluate_diagnostics(Ledger(tuple(broken), 10_000.0, 0.0), (),
                                 Ledger(rows, 10_000.0, 0.0))


if __name__ == "__main__":
    unittest.main()
