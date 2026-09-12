"""Strict validation wrapper for the DBMF managed-futures/CTA
replication ETF. Reuses qai_oos.run_scenario (same single-fund
leverage/financing model as QAI/DBV/SRRIX/MERFX) and the plain
5-year strict_validation gate — this experiment's sample (~7.3
years) does not support a stricter 10/15-year variant.
"""
import pandas as pd

import strict_validation
from qai_oos import run_scenario


def validate_prices(prices, maximum_gap_days=10):
    prices = pd.Series(prices).dropna().astype(float).sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate DBMF price date")
    if (prices <= 0).any():
        raise ValueError("DBMF adjusted prices must be positive")
    gaps = prices.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(f"Abnormally long DBMF price gap: {int(gaps.max())} days")
    return prices


def evaluate_five_year_oos(returns):
    return strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True
    )
