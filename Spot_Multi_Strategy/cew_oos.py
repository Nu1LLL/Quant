"""Strict validation wrapper for the CEW emerging-currency ETF."""
import strict_validation

from qai_oos import run_scenario, validate_prices


def evaluate_seventeen_year_oos(returns):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True, minimum_years=17.0
    )
    folds = result["walk_forward_folds"]
    result["checks"]["walk_forward_passed"] = bool(
        len(folds) >= 16 and not folds.empty and folds["fold_pass"].mean() >= 0.80
    )
    result["passed"] = all(result["checks"].values())
    return result
