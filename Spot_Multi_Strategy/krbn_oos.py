"""Strict validation wrapper for the KRBN carbon-allowance ETF."""
import strict_validation

from qai_oos import run_scenario, validate_prices


def evaluate_five_year_oos(returns):
    return strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True
    )
