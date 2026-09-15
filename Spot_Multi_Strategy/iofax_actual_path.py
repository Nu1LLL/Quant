"""Frozen account model for the IOFAX mortgage-credit NAV audit."""
import numpy as np
import pandas as pd

import cvsix_actual_path
import strict_validation


FROZEN_LEVERAGES = (1.0, 1.5, 2.0, 3.0, 5.0)


def validate_prices(prices, maximum_gap_days=10):
    prices = pd.Series(prices).dropna().astype(float).sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate IOFAX NAV date")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("IOFAX adjusted NAV must be finite and positive")
    gaps = prices.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(f"Abnormally long IOFAX NAV gap: {int(gaps.max())} days")
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
    if leverage not in FROZEN_LEVERAGES:
        raise ValueError("Leverage is outside the preregistered scenarios")
    for name, value in (
        ("entry_sales_load", entry_sales_load),
        ("exit_sales_load", exit_sales_load),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    gross = prices.pct_change(fill_method=None).dropna()
    external_cost = pd.Series(0.0, index=gross.index, name="external_cost")
    external_cost.iloc[0] = leverage * (leg_cost + entry_sales_load)
    external_cost.iloc[-1] += leverage * (leg_cost + exit_sales_load)
    financing_cost = pd.Series(
        max(leverage - 1.0, 0.0) * annual_financing / 252.0,
        index=gross.index,
        name="financing_cost",
    )
    net = leverage * gross - external_cost - financing_cost
    failure = np.flatnonzero(net.to_numpy() <= -1.0)
    if len(failure):
        first = int(failure[0])
        net.iloc[first] = -1.0
        if first + 1 < len(net):
            net.iloc[first + 1:] = 0.0
    net.name = "net_return"
    detail = pd.DataFrame({
        "gross_fund_return": gross,
        "leverage": leverage,
        "external_cost": external_cost,
        "financing_cost": financing_cost,
        "net_return": net,
    })
    detail["equity"] = initial_capital * (1.0 + net).cumprod()
    detail["solvent"] = detail["equity"] > 0.0
    return net, detail


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
    checks["sample_at_least_10y"] = elapsed_years >= 10.0
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
    returns, simulations=5000, block_length=63, seed=20150528,
    batch_size=100,
):
    return cvsix_actual_path.circular_block_monte_carlo_chunked(
        returns, simulations=simulations, block_length=block_length,
        seed=seed, batch_size=batch_size,
    )
