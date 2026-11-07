"""Frozen metrics for the preregistered BTC/ETH distance-pairs experiment.

This module consumes simulation records.  It never downloads market data,
forms signals, or selects parameters.  The only stochastic calculation is the
predeclared PCG64 circular 21-day block bootstrap.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
import math
from typing import Any, Iterable

import numpy as np


INITIAL_EQUITY = 10_000.0
OOS_START = date(2021, 1, 1)
OOS_END = date(2026, 1, 1)
OOS_DAYS = 1_826
YEARS = (2021, 2022, 2023, 2024, 2025)
REGIMES = ("bull_high", "bull_low", "bear_high", "bear_low")
SCENARIOS = (
    "base_1x",
    "base_2x",
    "doubled_cost_2x",
    "no_funding_credit_2x",
)
MC_PATHS = 5_000
MC_BLOCK_DAYS = 21
MC_SEED = 20261108


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _day_arrays(days: Iterable[Any], *, require_fixed_oos: bool) -> tuple[list[Any], np.ndarray]:
    """Validate the day boundary and the preregistered a/b/c/d compression."""
    records = list(days)
    if not records:
        raise ValueError("daily series is empty")
    if require_fixed_oos and len(records) != OOS_DAYS:
        raise ValueError("formal OOS must have exactly 1,826 daily records")

    factors = np.empty((len(records), 4), dtype=np.float64)
    previous_end = INITIAL_EQUITY
    previous_day = None
    for j, row in enumerate(records):
        day = row.date
        if isinstance(day, datetime) or not isinstance(day, date):
            raise ValueError("daily record date must be a UTC calendar date")
        if require_fixed_oos and day != OOS_START + timedelta(days=j):
            raise ValueError("formal OOS date is missing, duplicated, or out of order")
        if previous_day is not None and day != previous_day + timedelta(days=1):
            raise ValueError("daily records must be consecutive")
        start = _finite_positive(row.equity_start, "daily starting equity")
        end = _finite_positive(row.equity_end, "daily ending equity")
        if not math.isclose(start, previous_end, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("daily starting equity does not match previous ending equity")
        b = _finite_positive(row.max_regular_factor, "daily maximum regular factor")
        c = _finite_positive(row.min_adverse_factor, "daily minimum adverse factor")
        d = float(row.max_intraday_drawdown)
        a = end / start
        if b < max(1.0, a) - 1e-10:
            raise ValueError("daily regular maximum must include start and end")
        if c > min(1.0, a) + 1e-10:
            raise ValueError("daily adverse minimum must include start and end")
        if not math.isfinite(d) or not 0 <= d < 1:
            raise ValueError("daily intraday drawdown must be in [0, 1)")
        factors[j] = (a, b, c, d)
        previous_day, previous_end = day, end
    if require_fixed_oos and records[-1].date != OOS_END - timedelta(days=1):
        raise ValueError("formal OOS does not end on 2025-12-31")
    return records, factors


def _metrics_from_factors(factors: np.ndarray, *, initial_equity: float = INITIAL_EQUITY) -> dict:
    """Compound daily a factors and exactly replay minute-stress drawdown.

    The prior global regular-equity peak and the within-day regular peak
    cover the two possible denominators for every adverse minute mark.
    """
    equity = peak = _finite_positive(initial_equity, "initial equity")
    gains = losses = mean = m2 = max_drawdown = 0.0
    n = len(factors)
    if n == 0:
        raise ValueError("empty factor path")
    for j, (a, b, c, d) in enumerate(factors, start=1):
        start = equity
        equity = start * float(a)
        if not math.isfinite(equity) or equity <= 0:
            return {
                "ending_equity": equity,
                "cagr": -1.0,
                "sharpe": None,
                "max_drawdown": 1.0,
                "profit_factor": None,
            }
        pnl = equity - start
        if pnl > 0:
            gains += pnl
        elif pnl < 0:
            losses -= pnl
        daily_return = float(a) - 1.0
        delta = daily_return - mean
        mean += delta / j
        m2 += delta * (daily_return - mean)

        prior_peak = peak
        max_drawdown = max(
            max_drawdown,
            float(d),
            1.0 - start * float(c) / prior_peak,
        )
        peak = max(peak, start * float(b))

    sample_sd = math.sqrt(m2 / (n - 1)) if n > 1 and m2 > 0 else 0.0
    return {
        "ending_equity": equity,
        "cagr": (equity / initial_equity) ** (365.25 / n) - 1.0,
        "sharpe": math.sqrt(365.0) * mean / sample_sd if sample_sd > 0 else None,
        "max_drawdown": max_drawdown,
        "profit_factor": gains / losses if losses > 0 else None,
    }


def _passes_formal(metrics: dict) -> bool:
    return (
        metrics["cagr"] is not None
        and metrics["cagr"] >= 0.40
        and metrics["sharpe"] is not None
        and metrics["sharpe"] > 1.5
        and metrics["max_drawdown"] <= 0.20
        and metrics["profit_factor"] is not None
        and metrics["profit_factor"] > 1.3
    )


def _passes_signal_quality(metrics: dict) -> bool:
    return (
        metrics["cagr"] is not None
        and metrics["cagr"] > 0
        and metrics["sharpe"] is not None
        and metrics["sharpe"] > 1.5
        and metrics["max_drawdown"] <= 0.20
        and metrics["profit_factor"] is not None
        and metrics["profit_factor"] > 1.3
    )


def monte_carlo_compressed(
    factors: np.ndarray,
    *,
    paths: int = MC_PATHS,
    block_days: int = MC_BLOCK_DAYS,
    seed: int = MC_SEED,
) -> dict:
    """Run the pinned PCG64 circular block replay on a/b/c/d day records."""
    values = np.asarray(factors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0:
        raise ValueError("factors must have shape (days, 4)")
    if paths <= 0 or block_days <= 0:
        raise ValueError("paths and block_days must be positive")
    n = len(values)
    n_blocks = (n + block_days - 1) // block_days
    rng = np.random.Generator(np.random.PCG64(seed))
    offsets = np.arange(block_days, dtype=np.int64)
    collected = {key: [] for key in ("cagr", "sharpe", "max_drawdown", "profit_factor")}
    undefined = {key: 0 for key in collected}
    joint_pass_count = 0
    for _ in range(paths):
        starts = rng.integers(0, n, size=n_blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        metrics = _metrics_from_factors(values[indices])
        joint_pass_count += _passes_formal(metrics)
        for key in collected:
            value = metrics[key]
            if value is None or not math.isfinite(value):
                undefined[key] += 1
            else:
                collected[key].append(value)

    percentiles = {}
    for key, scores in collected.items():
        if scores:
            p05, p50, p95 = np.percentile(scores, [5, 50, 95])
            percentiles[key] = {"p05": float(p05), "p50": float(p50), "p95": float(p95)}
        else:
            percentiles[key] = {"p05": None, "p50": None, "p95": None}
    return {
        "paths": paths,
        "block_days": block_days,
        "seed": seed,
        "rng": "numpy.random.Generator(numpy.random.PCG64)",
        "joint_pass_count": joint_pass_count,
        "joint_pass_fraction": joint_pass_count / paths,
        "percentiles": percentiles,
        "undefined_counts": undefined,
    }


def _annual_and_monthly(records: list[Any], trades: list[Any]) -> tuple[dict, dict, dict]:
    entered = defaultdict(int)
    for trade in trades:
        entry = trade.entry_fill_time
        if not isinstance(entry, datetime) or entry.tzinfo is None or entry.utcoffset() != timedelta(0):
            raise ValueError("trade entry fill time must be a UTC aware datetime")
        entered[entry.year] += 1

    annual = {}
    for year in YEARS:
        subset = [r for r in records if r.date.year == year]
        if len(subset) not in (365, 366):
            raise ValueError("annual fold is incomplete")
        returns = np.array([r.equity_end / r.equity_start - 1.0 for r in subset])
        sd = float(np.std(returns, ddof=1))
        sharpe = math.sqrt(365.0) * float(np.mean(returns)) / sd if sd > 0 else None
        net_return = subset[-1].equity_end / subset[0].equity_start - 1.0
        count = entered[year]
        annual[str(year)] = {
            "entered_trades": count,
            "net_return": net_return,
            "daily_sharpe": sharpe,
            "pass": count >= 1 and net_return > 0 and sharpe is not None and sharpe > 0,
        }

    by_month = defaultdict(list)
    for row in records:
        by_month[row.date.strftime("%Y-%m")].append(row)
    monthly = {
        month: group[-1].equity_end / group[0].equity_start - 1.0
        for month, group in sorted(by_month.items())
    }
    if len(monthly) != 60:
        raise ValueError("monthly accounting must cover exactly 60 months")
    counts = {"positive": 0, "negative": 0, "flat": 0}
    losing_streak = longest_losing_streak = 0
    for value in monthly.values():
        if value > 0:
            counts["positive"] += 1
            losing_streak = 0
        elif value < 0:
            counts["negative"] += 1
            losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, losing_streak)
        else:
            counts["flat"] += 1
            losing_streak = 0
    month_summary = {
        **counts,
        "worst_month": min(monthly, key=monthly.get),
        "worst_month_return": min(monthly.values()),
        "longest_losing_month_streak": longest_losing_streak,
    }
    return annual, monthly, month_summary


def _regime_report(trades: list[Any]) -> tuple[dict, bool]:
    buckets = {key: {"trades": 0, "net_pnl": 0.0} for key in REGIMES}
    buckets["unknown"] = {"trades": 0, "net_pnl": 0.0}
    for trade in trades:
        pnl = float(trade.net_pnl)
        if not math.isfinite(pnl):
            raise ValueError("trade net PnL must be finite")
        key = trade.regime if trade.regime in REGIMES else "unknown"
        buckets[key]["trades"] += 1
        buckets[key]["net_pnl"] += pnl
    passed = (
        buckets["unknown"]["trades"] == 0
        and all(buckets[key]["trades"] > 0 and buckets[key]["net_pnl"] > 0 for key in REGIMES)
    )
    return buckets, passed


def evaluate_gate(scenarios: dict[str, Any]) -> dict:
    """Evaluate the frozen five-year Gate; executable-data evidence stays unpassed."""
    if set(scenarios) != set(SCENARIOS):
        raise ValueError(f"four fixed scenarios required: {SCENARIOS}")
    records = {}
    factors = {}
    metrics = {}
    replay_verified = {}
    for name in SCENARIOS:
        scenario = scenarios[name]
        records[name], factors[name] = _day_arrays(scenario.daily_records, require_fixed_oos=True)
        metrics[name] = _metrics_from_factors(factors[name])
        simulated_dd = float(scenario.max_drawdown)
        replay_verified[name] = math.isclose(
            metrics[name]["max_drawdown"], simulated_dd, rel_tol=1e-8, abs_tol=1e-9
        )
        if not replay_verified[name]:
            raise ValueError(f"compressed replay does not reproduce {name} minute drawdown")

    base_trades = list(scenarios["base_2x"].trades)
    def schedule(trades: Iterable[Any]) -> list[tuple]:
        return [
            (t.entry_signal_time, t.entry_fill_time, t.exit_fill_time, t.direction)
            for t in trades
        ]

    base_schedule = schedule(base_trades)
    for name in SCENARIOS:
        if schedule(scenarios[name].trades) != base_schedule:
            raise ValueError(f"{name} does not use the frozen common pair-trade schedule")
    annual, monthly, month_summary = _annual_and_monthly(records["base_2x"], base_trades)
    regimes, regimes_pass = _regime_report(base_trades)
    mc = monte_carlo_compressed(factors["base_2x"])
    gate = {
        "signal_quality_1x": _passes_signal_quality(metrics["base_1x"]),
        "formal_2x": _passes_formal(metrics["base_2x"]),
        "annual_walk_forward": all(fold["pass"] for fold in annual.values()),
        "four_market_states": regimes_pass,
        "monte_carlo": mc["joint_pass_fraction"] >= 0.80,
        "doubled_nonfee_cost_profitable": (
            metrics["doubled_cost_2x"]["ending_equity"] > INITIAL_EQUITY
        ),
        "no_funding_credit_profitable": (
            metrics["no_funding_credit_2x"]["ending_equity"] > INITIAL_EQUITY
        ),
        "independent_executable_data": False,
    }
    gate["all_numeric"] = all(value for key, value in gate.items() if key != "independent_executable_data")
    gate["full"] = gate["all_numeric"] and gate["independent_executable_data"]
    return {
        "metrics": metrics,
        "formal_cagr_within_40_to_60_percent_target_band": (
            0.40 <= metrics["base_2x"]["cagr"] <= 0.60
        ),
        "annual_folds": annual,
        "monthly_returns": monthly,
        "monthly_summary": month_summary,
        "completed_pair_round_trips": len(base_trades),
        "market_states": regimes,
        "monte_carlo": mc,
        "compressed_replay_verified": replay_verified,
        "gate": gate,
        "evidence_grade": "retrospective pair-mechanism-internal blind historical holdout",
    }
