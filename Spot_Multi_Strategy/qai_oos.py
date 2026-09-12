"""Fixed-cost leveraged return model for the QAI multi-strategy ETF."""
import pandas as pd

import strict_validation


def validate_prices(prices, maximum_gap_days=10):
    prices = pd.Series(prices).dropna().astype(float).sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate QAI price date")
    if (prices <= 0).any():
        raise ValueError("QAI adjusted prices must be positive")
    gaps = prices.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(f"Abnormally long QAI price gap: {int(gaps.max())} days")
    return prices


def run_scenario(prices, leverage, leg_cost=0.0005, annual_financing=0.04):
    prices = validate_prices(prices)
    gross = prices.pct_change(fill_method=None).dropna()
    turnover = leverage * gross.abs()
    trading_cost = turnover * leg_cost
    trading_cost.iloc[0] += leverage * leg_cost
    trading_cost.iloc[-1] += leverage * leg_cost
    financing = max(leverage - 1.0, 0.0) * annual_financing / 252.0
    net = leverage * gross - trading_cost - financing
    net.name = "net_return"
    detail = pd.DataFrame({
        "gross_fund_return": gross,
        "leverage": leverage,
        "turnover": turnover,
        "trading_cost": trading_cost,
        "financing_cost": financing,
        "net_return": net,
    })
    return net, detail


def evaluate_ten_year_oos(returns):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True
    )
    folds = result["walk_forward_folds"]
    ten_year_walk_forward = bool(
        len(folds) >= 10 and len(folds) > 0 and folds["fold_pass"].mean() >= 0.80
    )
    elapsed_years = result["metrics"]["elapsed_days"] / 365.25
    checks = dict(result["checks"])
    checks.pop("sample_at_least_5y", None)
    checks["sample_at_least_10y"] = elapsed_years >= 10.0
    checks["walk_forward_passed"] = ten_year_walk_forward
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result
