"""Strict validation wrapper for the DBV currency-carry ETF."""
import strict_validation

from qai_oos import run_scenario, validate_prices


def evaluate_fifteen_year_oos(returns):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True
    )
    folds = result["walk_forward_folds"]
    long_walk_forward = bool(
        len(folds) >= 15 and len(folds) > 0 and folds["fold_pass"].mean() >= 0.80
    )
    elapsed_years = result["metrics"]["elapsed_days"] / 365.25
    checks = dict(result["checks"])
    checks.pop("sample_at_least_5y", None)
    checks["sample_at_least_15y"] = elapsed_years >= 15.0
    checks["walk_forward_passed"] = long_walk_forward
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result
