"""Cboe PTLT Treasury putwrite strict OOS helpers."""
from pathlib import Path

import numpy as np
import pandas as pd

from cboe_options_benchmark import load_index


LATEST_START = pd.Timestamp("2010-12-31", tz="UTC")
EARLIEST_END = pd.Timestamp("2026-08-31", tz="UTC")


def load_levels(cache_dir, refresh=False):
    return load_index("PTLT", Path(cache_dir), refresh=refresh).to_frame()


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
        and audit["span_years"] >= 15.0 and audit["coverage_ratio"] >= 0.95
        and audit["max_business_gap"] <= 10 and audit["duplicate_count"] == 0
        and not levels.isna().any().any() and (levels > 0).all().all()
    )
    return audit


def base_returns(levels, annual_implementation_drag=0.015, endpoint_cost=0.001):
    returns = levels["PTLT"].pct_change(fill_method=None).fillna(0.0)
    returns = returns - annual_implementation_drag / 252.0
    returns.iloc[0] -= endpoint_cost
    returns.iloc[-1] -= endpoint_cost
    return returns.rename("net_return")


def apply_leverage(returns, leverage, annual_financing=0.04):
    result = pd.Series(returns).copy() * leverage
    result -= max(leverage - 1.0, 0.0) * annual_financing / 252.0
    bankrupt = result.le(-1.0)
    if bankrupt.any():
        first = bankrupt[bankrupt].index[0]
        result.loc[first] = -1.0
        result.loc[result.index > first] = 0.0
    return result
