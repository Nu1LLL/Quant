"""Cboe CNDR/BFLY fixed-weight options-premium blend."""
from pathlib import Path

import numpy as np
import pandas as pd

from cboe_options_benchmark import load_index


SYMBOLS = ("CNDR", "BFLY")
LATEST_START = pd.Timestamp("1990-12-31", tz="UTC")
EARLIEST_END = pd.Timestamp("2026-08-31", tz="UTC")


def load_levels(cache_dir, refresh=False):
    series = [load_index(symbol, Path(cache_dir), refresh=refresh) for symbol in SYMBOLS]
    return pd.concat(series, axis=1, join="inner").dropna().sort_index()


def coverage_audit(levels):
    if levels.empty:
        return {"observations": 0, "start": None, "end": None,
                "coverage_ratio": 0.0, "max_business_gap": np.inf,
                "duplicate_count": 0, "span_years": 0.0, "passed": False}
    index = pd.DatetimeIndex(levels.index).tz_convert("UTC")
    expected = pd.bdate_range(index[0].normalize(), index[-1].normalize(), tz="UTC")
    locations = expected.get_indexer(index.normalize())
    max_gap = int(np.diff(locations).max() - 1) if len(locations) > 1 else 0
    audit = {
        "observations": int(len(index)), "start": index[0], "end": index[-1],
        "coverage_ratio": float(len(index.intersection(expected)) / len(expected)),
        "max_business_gap": max_gap, "duplicate_count": int(index.duplicated().sum()),
        "span_years": float((index[-1] - index[0]).days / 365.25),
    }
    audit["passed"] = bool(
        index[0] <= LATEST_START and index[-1] >= EARLIEST_END
        and audit["span_years"] >= 30.0 and audit["coverage_ratio"] >= 0.95
        and audit["max_business_gap"] <= 10 and audit["duplicate_count"] == 0
        and not levels.isna().any().any() and (levels > 0).all().all()
    )
    return audit


def blend_returns(levels, rebalance_cost=0.001, annual_implementation_drag=0.02):
    """Monthly 50/50 rebalancing using weights fixed before each daily return."""
    returns = levels.loc[:, SYMBOLS].pct_change(fill_method=None).fillna(0.0)
    target = np.array([0.5, 0.5])
    weights = target.copy()
    values, turnovers = [], []
    months = returns.index.tz_localize(None).to_period("M")
    for row_number, row in enumerate(returns.to_numpy(dtype=float)):
        rebalance = row_number == 0 or months[row_number] != months[row_number - 1]
        turnover = float(np.abs(target - weights).sum()) if rebalance else 0.0
        if row_number == 0:
            turnover = 1.0
        if rebalance:
            weights = target.copy()
        gross = float(np.dot(weights, row))
        net = gross - turnover * rebalance_cost - annual_implementation_drag / 252.0
        values.append(net)
        turnovers.append(turnover)
        denominator = 1.0 + gross
        if denominator > 0:
            weights = weights * (1.0 + row) / denominator
    turnovers[-1] += float(np.abs(weights).sum())
    values[-1] -= float(np.abs(weights).sum()) * rebalance_cost
    return (pd.Series(values, index=returns.index, name="iron_blend_return"),
            pd.Series(turnovers, index=returns.index, name="turnover"))


def apply_leverage(base_returns, leverage, annual_financing=0.04):
    raw = pd.Series(base_returns) * leverage
    raw -= max(leverage - 1.0, 0.0) * annual_financing / 252.0
    bankrupt = raw.le(-1.0)
    if bankrupt.any():
        first = bankrupt[bankrupt].index[0]
        raw.loc[first] = -1.0
        raw.loc[raw.index > first] = 0.0
    return raw
