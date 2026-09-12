"""Strict validation wrapper for the PFIX long-rate-convexity ETF."""
import strict_validation

from qai_oos import run_scenario as _run_scenario
from qai_oos import validate_prices


def run_scenario(prices, leverage):
    return _run_scenario(prices, leverage, leg_cost=0.0015, annual_financing=0.04)


def evaluate_five_year_oos(returns):
    validation = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True
    )
    folds = validation["walk_forward_folds"]
    walk_forward_passed = (
        len(folds) >= 4
        and not folds.empty
        and float(folds["fold_pass"].mean()) >= 0.80
    )
    validation["checks"]["walk_forward_passed"] = walk_forward_passed
    validation["passed"] = all(validation["checks"].values())
    return validation
