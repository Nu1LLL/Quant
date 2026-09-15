"""Frozen account model for the QSPNX actual multi-strategy NAV audit."""
import numpy as np
import pandas as pd

import strict_validation


def validate_prices(prices, maximum_gap_days=10):
    prices = pd.Series(prices).dropna().astype(float).sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate QSPNX NAV date")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("QSPNX adjusted NAV must be finite and positive")
    gaps = prices.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(f"Abnormally long QSPNX NAV gap: {int(gaps.max())} days")
    return prices


def run_scenario(
    prices, leverage, leg_cost=0.0010, annual_financing=0.04,
    initial_capital=10000.0,
):
    """Apply fixed external leverage to adjusted NAV returns.

    QSPNX NAV already contains fund-level expenses and internal trading.  The
    external account therefore pays only one entry and one exit cost, plus
    financing on exposure above 1x.
    """
    prices = validate_prices(prices)
    if leverage not in (1.0, 2.0, 3.0, 4.0, 5.0):
        raise ValueError("Leverage must be one of the five preregistered scenarios")
    gross = prices.pct_change(fill_method=None).dropna()
    external_cost = pd.Series(0.0, index=gross.index, name="external_cost")
    external_cost.iloc[0] += leverage * leg_cost
    external_cost.iloc[-1] += leverage * leg_cost
    financing_cost = pd.Series(
        max(leverage - 1.0, 0.0) * annual_financing / 252.0,
        index=gross.index,
        name="financing_cost",
    )
    net = leverage * gross - external_cost - financing_cost
    if (net <= -1.0).any():
        first_failure = net.index[net <= -1.0][0]
        failure_location = net.index.get_loc(first_failure)
        net.iloc[failure_location] = -1.0
        if failure_location + 1 < len(net):
            net.iloc[failure_location + 1:] = 0.0
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


def evaluate_actual_path(returns, solvent):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=False, costs_included=True
    )
    folds = result["walk_forward_folds"]
    long_walk_forward = bool(
        len(folds) >= 10 and len(folds) > 0 and folds["fold_pass"].mean() >= 0.80
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
    checks["sample_at_least_12_5y"] = elapsed_years >= 12.5
    checks["walk_forward_passed"] = long_walk_forward
    checks["worst_rolling_3y_sharpe_at_least_1"] = worst_rolling >= 1.0
    checks["solvent"] = bool(solvent)
    result["checks"] = checks
    result["metrics"]["worst_rolling_3y_sharpe"] = worst_rolling
    result["passed"] = all(checks.values())
    return result
