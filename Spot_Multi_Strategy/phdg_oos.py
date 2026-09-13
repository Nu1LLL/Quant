"""Buy-and-hold PHDG downside-hedged ETF model with external leverage."""
import numpy as np
import pandas as pd


SYMBOL = "PHDG"
LATEST_ACCEPTABLE_START = pd.Timestamp("2013-01-31", tz="UTC")


def validate_prices(prices):
    series = pd.Series(prices, name=SYMBOL).copy().sort_index().dropna()
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")
    values = series.to_numpy(dtype=float)
    if series.empty or series.index.has_duplicates or not np.isfinite(values).all():
        raise ValueError("PHDG series is empty, duplicated or non-finite")
    if (values <= 0).any():
        raise ValueError("PHDG prices must be positive")
    gaps = series.index.to_series().diff().dropna().dt.days
    if not gaps.empty and int(gaps.max()) > 10:
        raise ValueError("PHDG series has a gap longer than ten days")
    return series


def apply_bankruptcy(raw_returns):
    reported = pd.Series(raw_returns).copy()
    bankrupt = reported.le(-1.0)
    if bankrupt.any():
        first = bankrupt[bankrupt].index[0]
        reported.loc[first] = -1.0
        reported.loc[reported.index > first] = 0.0
    return reported


def run_scenario(prices, leverage, transaction_cost=0.001, annual_financing=0.04):
    prices = validate_prices(prices)
    target = pd.Series(leverage, index=prices.index, name="target")
    position = target.shift(2).fillna(0.0)
    asset_return = prices.pct_change(fill_method=None)
    gross_return = position * asset_return
    turnover = position.diff().abs().fillna(0.0)
    trading_cost = turnover * transaction_cost
    if position.iloc[-1] > 0:
        trading_cost.iloc[-1] += position.iloc[-1] * transaction_cost
    active = position.gt(0).astype(float)
    financing = active * max(leverage - 1.0, 0.0) * annual_financing / 252.0
    raw_net_return = gross_return - trading_cost - financing
    net_return = apply_bankruptcy(raw_net_return)
    detail = pd.DataFrame({
        "price": prices, "position": position, "gross_return": gross_return,
        "turnover": turnover, "trading_cost": trading_cost,
        "financing_cost": financing, "raw_net_return": raw_net_return,
        "net_return": net_return,
    }).dropna(subset=["net_return"])
    detail["equity"] = 10000.0 * (1.0 + detail["net_return"]).cumprod()
    detail["bankrupt"] = detail["equity"].eq(0.0)
    return detail["net_return"], detail, position.loc[detail.index]
