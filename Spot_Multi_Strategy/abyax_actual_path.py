"""Frozen account model for the ABYAX multi-manager CTA NAV audit."""
import numpy as np
import pandas as pd

import cvsix_actual_path
import iofax_actual_path
import strict_validation


FROZEN_LEVERAGES = (1.0, 1.5, 2.0, 3.0, 5.0)


def validate_prices(prices, maximum_gap_days=10):
    prices = pd.Series(prices).dropna().astype(float).sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate ABYAX NAV date")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("ABYAX adjusted NAV must be finite and positive")
    gaps = prices.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(f"Abnormally long ABYAX NAV gap: {int(gaps.max())} days")
    return prices


def nav_quality(prices):
    gross = validate_prices(prices).pct_change(fill_method=None).dropna()
    zero_ratio = float((gross == 0.0).mean())
    lag1 = float(gross.autocorr(lag=1)) if len(gross) > 2 else np.nan
    return {
        "zero_return_ratio": zero_ratio,
        "lag1_return_autocorrelation": lag1,
        "unsmoothed_daily_nav": bool(
            zero_ratio <= 0.10 and np.isfinite(lag1) and abs(lag1) <= 0.30
        ),
    }


def run_scenario(
    prices, leverage, entry_sales_load=0.0, exit_sales_load=0.0,
    leg_cost=0.0010, annual_financing=0.04, initial_capital=10000.0,
):
    prices = validate_prices(prices)
    return iofax_actual_path.run_scenario(
        prices, leverage, entry_sales_load=entry_sales_load,
        exit_sales_load=exit_sales_load, leg_cost=leg_cost,
        annual_financing=annual_financing, initial_capital=initial_capital,
    )


def evaluate_actual_path(
    returns, solvent, account_executable, strategy_continuity,
    unsmoothed_daily_nav,
):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True
    )
    folds = result["walk_forward_folds"]
    walk_forward = bool(
        len(folds) >= 8 and len(folds) > 0 and folds["fold_pass"].mean() >= 0.80
    )
    elapsed_years = result["metrics"]["elapsed_days"] / 365.25
    rolling_mean = returns.rolling(756, min_periods=756).mean()
    rolling_std = returns.rolling(756, min_periods=756).std(ddof=0)
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252.0)
    worst_rolling = (
        float(rolling_sharpe.min()) if rolling_sharpe.notna().any() else np.nan
    )
    checks = dict(result["checks"])
    checks.pop("sample_at_least_5y", None)
    checks["sample_at_least_9_5y"] = elapsed_years >= 9.5
    checks["walk_forward_passed"] = walk_forward
    checks["worst_rolling_3y_sharpe_at_least_1"] = worst_rolling >= 1.0
    checks["account_executable_for_10000"] = bool(account_executable)
    checks["same_strategy_for_full_sample"] = bool(strategy_continuity)
    checks["unsmoothed_daily_nav"] = bool(unsmoothed_daily_nav)
    checks["solvent"] = bool(solvent)
    result["checks"] = checks
    result["metrics"]["worst_rolling_3y_sharpe"] = worst_rolling
    result["passed"] = all(checks.values())
    return result


def circular_block_monte_carlo_chunked(
    returns, simulations=5000, block_length=63, seed=20140701,
    batch_size=100,
):
    return cvsix_actual_path.circular_block_monte_carlo_chunked(
        returns, simulations=simulations, block_length=block_length,
        seed=seed, batch_size=batch_size,
    )
