"""Long-history four-asset 50/200 trend replication, isolated from core strategy."""
import numpy as np
import pandas as pd

import golden_cross_oos as golden_cross


SYMBOLS = ("MDY", "UDN", "DBA", "EMB")
LEG_COST = 0.001
LATEST_START = pd.Timestamp("2008-01-31", tz="UTC")
EARLIEST_END = pd.Timestamp("2026-08-31", tz="UTC")


def align_prices(prices_by_symbol):
    normalized = {}
    for symbol in SYMBOLS:
        if symbol not in prices_by_symbol:
            raise ValueError(f"Missing fixed symbol {symbol}")
        series = pd.Series(prices_by_symbol[symbol]).dropna().astype(float).sort_index()
        index = pd.DatetimeIndex(series.index)
        index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        index = index.normalize()
        if index.has_duplicates:
            raise ValueError(f"Duplicate calendar-date price for {symbol}")
        if (series <= 0).any():
            raise ValueError(f"Nonpositive price for {symbol}")
        normalized[symbol] = pd.Series(series.to_numpy(), index=index, name=symbol)
    return pd.concat(normalized.values(), axis=1, join="inner").dropna().sort_index()


def coverage_audit(prices):
    if prices.empty:
        return {"observations": 0, "start": None, "end": None,
                "coverage_ratio": 0.0, "max_business_gap": np.inf,
                "duplicate_count": 0, "span_years": 0.0, "passed": False}
    index = pd.DatetimeIndex(prices.index).tz_convert("UTC")
    expected = pd.bdate_range(index[0], index[-1], tz="UTC")
    locations = expected.get_indexer(index)
    max_gap = int(np.diff(locations).max() - 1) if len(locations) > 1 else 0
    audit = {
        "observations": int(len(index)), "start": index[0], "end": index[-1],
        "coverage_ratio": float(len(index.intersection(expected)) / len(expected)),
        "max_business_gap": max_gap, "duplicate_count": int(index.duplicated().sum()),
        "span_years": float((index[-1] - index[0]).days / 365.25),
    }
    audit["passed"] = bool(
        index[0] <= LATEST_START and index[-1] >= EARLIEST_END
        and audit["span_years"] >= 18.0 and audit["coverage_ratio"] >= 0.94
        and audit["max_business_gap"] <= 10 and audit["duplicate_count"] == 0
        and not prices.isna().any().any() and (prices > 0).all().all()
    )
    return audit


def run_sleeve(prices, leverage, leg_cost=LEG_COST, annual_financing=0.04):
    prices = golden_cross.validate_prices(prices)
    full_position = golden_cross.build_target_positions(prices, leverage)
    returns = prices.pct_change(fill_method=None).dropna()
    position = full_position.reindex(returns.index).fillna(0.0)
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    trading_cost = turnover * leg_cost
    trading_cost.iloc[-1] += abs(position.iloc[-1]) * leg_cost
    financing = ((position > 1.0).astype(float) * max(leverage - 1.0, 0.0)
                 * annual_financing / 252.0)
    net = position * returns - trading_cost - financing
    detail = pd.DataFrame({"asset_return": returns, "position": position,
                           "turnover": turnover, "trading_cost": trading_cost,
                           "financing_cost": financing, "net_return": net})
    return net.rename("net_return"), detail


def run_portfolio(prices, leverage):
    sleeve_returns, details = {}, {}
    for symbol in SYMBOLS:
        sleeve_returns[symbol], details[symbol] = run_sleeve(prices[symbol], leverage)
    sleeves = pd.DataFrame(sleeve_returns).dropna()
    combined = sleeves.mean(axis=1).rename("net_return")
    bankrupt = combined.le(-1.0)
    if bankrupt.any():
        first = bankrupt[bankrupt].index[0]
        combined.loc[first] = -1.0
        combined.loc[combined.index > first] = 0.0
    return combined, sleeves, details
