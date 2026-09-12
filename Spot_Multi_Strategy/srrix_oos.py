"""Strict validation wrapper for the SRRIX reinsurance-risk-premium fund."""
import pandas as pd

import strict_validation
from qai_oos import run_scenario


def validate_prices(prices, maximum_gap_days=10):
    prices = pd.Series(prices).dropna().astype(float).sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate SRRIX price date")
    if (prices <= 0).any():
        raise ValueError("SRRIX adjusted prices must be positive")
    gaps = prices.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(f"Abnormally long SRRIX price gap: {int(gaps.max())} days")
    return prices


def nav_quality(returns):
    returns = pd.Series(returns).dropna().astype(float).sort_index()
    zero_return_ratio = float((returns == 0.0).mean())
    lag1_autocorrelation = float(returns.autocorr(lag=1))
    return {
        "zero_return_ratio": zero_return_ratio,
        "lag1_autocorrelation": lag1_autocorrelation,
        "zero_return_ratio_at_most_10pct": zero_return_ratio <= 0.10,
        "absolute_lag1_autocorrelation_at_most_0_20": (
            pd.notna(lag1_autocorrelation) and abs(lag1_autocorrelation) <= 0.20
        ),
    }


def evaluate_ten_year_oos(returns, nav_returns=None):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True
    )
    folds = result["walk_forward_folds"]
    long_walk_forward = bool(
        len(folds) >= 10 and len(folds) > 0 and folds["fold_pass"].mean() >= 0.80
    )
    elapsed_years = result["metrics"]["elapsed_days"] / 365.25
    quality = nav_quality(returns if nav_returns is None else nav_returns)
    checks = dict(result["checks"])
    checks.pop("sample_at_least_5y", None)
    checks["sample_at_least_10y"] = elapsed_years >= 10.0
    checks["walk_forward_passed"] = long_walk_forward
    checks["zero_return_ratio_at_most_10pct"] = quality[
        "zero_return_ratio_at_most_10pct"
    ]
    checks["absolute_lag1_autocorrelation_at_most_0_20"] = quality[
        "absolute_lag1_autocorrelation_at_most_0_20"
    ]
    result["checks"] = checks
    result["nav_quality"] = quality
    result["passed"] = all(checks.values())
    return result
