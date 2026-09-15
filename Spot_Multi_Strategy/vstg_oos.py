"""Frozen Cboe VSTG volatility-premium account model."""
from pathlib import Path

import numpy as np
import pandas as pd

from cboe_options_benchmark import load_index
import cvsix_actual_path
import strict_validation


SYMBOL = "VSTG"
FROZEN_LEVERAGES = (1.0, 1.5, 2.0, 3.0, 5.0)
LATEST_START = pd.Timestamp("2010-12-31", tz="UTC")
EARLIEST_END = pd.Timestamp("2026-08-31", tz="UTC")


def load_levels(cache_dir, refresh=False):
    return load_index(SYMBOL, Path(cache_dir), refresh=refresh)


def validate_levels(levels, maximum_gap_days=10):
    levels = pd.Series(levels).dropna().astype(float).sort_index()
    if levels.index.has_duplicates:
        raise ValueError("Duplicate VSTG index date")
    if not np.isfinite(levels).all() or (levels <= 0).any():
        raise ValueError("VSTG index levels must be finite and positive")
    gaps = levels.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(f"Abnormally long VSTG history gap: {int(gaps.max())} days")
    return levels


def coverage_audit(levels):
    levels = validate_levels(levels)
    weekdays = pd.date_range(levels.index[0], levels.index[-1], freq="B", tz="UTC")
    coverage = len(levels.index.normalize().unique()) / len(weekdays)
    audit = {
        "observations": int(len(levels)),
        "start": levels.index[0],
        "end": levels.index[-1],
        "span_years": float((levels.index[-1] - levels.index[0]).days / 365.25),
        "weekday_coverage": float(coverage),
        "start_by_2010_12_31": bool(levels.index[0] <= LATEST_START),
        "end_by_2026_08_31": bool(levels.index[-1] >= EARLIEST_END),
        "span_at_least_15y": bool(
            (levels.index[-1] - levels.index[0]).days / 365.25 >= 15.0
        ),
        "weekday_coverage_at_least_94pct": bool(coverage >= 0.94),
    }
    audit["passed"] = all(
        audit[name] for name in (
            "start_by_2010_12_31", "end_by_2026_08_31",
            "span_at_least_15y", "weekday_coverage_at_least_94pct",
        )
    )
    return audit


def run_scenario(
    levels, leverage, annual_implementation_drag=0.02,
    annual_financing=0.04, entry_exit_cost=0.0010,
    initial_capital=10000.0,
):
    levels = validate_levels(levels)
    if leverage not in FROZEN_LEVERAGES:
        raise ValueError("Leverage is outside the preregistered scenarios")
    gross = levels.pct_change(fill_method=None).dropna()
    implementation = pd.Series(
        leverage * annual_implementation_drag / 252.0,
        index=gross.index, name="implementation_drag",
    )
    financing = pd.Series(
        max(leverage - 1.0, 0.0) * annual_financing / 252.0,
        index=gross.index, name="financing_cost",
    )
    external = pd.Series(0.0, index=gross.index, name="external_cost")
    external.iloc[0] = leverage * entry_exit_cost
    external.iloc[-1] += leverage * entry_exit_cost
    net = leverage * gross - implementation - financing - external
    failure = np.flatnonzero(net.to_numpy() <= -1.0)
    if len(failure):
        first = int(failure[0])
        net.iloc[first] = -1.0
        if first + 1 < len(net):
            net.iloc[first + 1:] = 0.0
    net.name = "net_return"
    detail = pd.DataFrame({
        "gross_index_return": gross,
        "leverage": leverage,
        "implementation_drag": implementation,
        "financing_cost": financing,
        "external_cost": external,
        "net_return": net,
    })
    detail["equity"] = initial_capital * (1.0 + net).cumprod()
    detail["solvent"] = detail["equity"] > 0.0
    return net, detail


def evaluate(
    returns, solvent, account_executable, live_history_at_least_10y,
):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True, minimum_years=15.0
    )
    folds = result["walk_forward_folds"]
    result["checks"]["walk_forward_passed"] = bool(
        len(folds) >= 12 and not folds.empty and folds["fold_pass"].mean() >= 0.80
    )
    result["checks"]["worst_rolling_3y_sharpe_at_least_1"] = bool(
        result["metrics"]["worst_rolling_3y_sharpe"] >= 1.0
    )
    result["checks"]["account_executable_for_10000"] = bool(account_executable)
    result["checks"]["live_history_at_least_10y"] = bool(live_history_at_least_10y)
    result["checks"]["solvent"] = bool(solvent)
    result["passed"] = all(result["checks"].values())
    return result


def circular_block_monte_carlo_chunked(
    returns, simulations=5000, block_length=21, seed=20080616,
    batch_size=100,
):
    return cvsix_actual_path.circular_block_monte_carlo_chunked(
        returns, simulations=simulations, block_length=block_length,
        seed=seed, batch_size=batch_size,
    )
