"""Strict validation wrapper for the BDRY dry-bulk freight futures ETF."""
import strict_validation

from qai_oos import run_scenario as _run_scenario
from qai_oos import validate_prices


def run_scenario(prices, leverage):
    return _run_scenario(prices, leverage, leg_cost=0.0010, annual_financing=0.04)


def evaluate_five_year_oos(returns):
    return strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True
    )
