"""Causal monthly 200-day trend model for Dollar Index futures."""
import numpy as np
import pandas as pd


SYMBOL = "DX-Y.NYB"
LATEST_ACCEPTABLE_START = pd.Timestamp("2001-01-31", tz="UTC")


def validate_prices(prices):
    series = pd.Series(prices, name=SYMBOL).copy().sort_index().dropna()
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")
    values = series.to_numpy(dtype=float)
    if series.empty or series.index.has_duplicates or not np.isfinite(values).all():
        raise ValueError("Dollar Index futures series is empty, duplicated or non-finite")
    if (values <= 0).any():
        raise ValueError("Dollar Index futures prices must be positive")
    gaps = series.index.to_series().diff().dropna().dt.days
    if not gaps.empty and int(gaps.max()) > 10:
        raise ValueError("Dollar Index futures series has a gap longer than ten days")
    return series


def causal_positions(prices, window=200):
    prices = validate_prices(prices)
    average = prices.rolling(window, min_periods=window).mean()
    signal = prices.gt(average).map({True: 1.0, False: -1.0}).where(average.notna())
    month = pd.Series(prices.index.year * 100 + prices.index.month, index=prices.index)
    month_end = month.ne(month.shift(-1))
    targets = signal.where(month_end).ffill().fillna(0.0)
    positions = targets.shift(2).fillna(0.0)
    return positions, average, targets


def apply_bankruptcy(raw_returns):
    reported = pd.Series(raw_returns).copy()
    bankrupt = reported.le(-1.0)
    if bankrupt.any():
        first = bankrupt[bankrupt].index[0]
        reported.loc[first] = -1.0
        reported.loc[reported.index > first] = 0.0
    return reported


def run_scenario(
    prices, leverage, transaction_cost=0.0005, annual_roll_haircut=0.01,
    annual_financing=0.04,
):
    prices = validate_prices(prices)
    base_position, average, targets = causal_positions(prices)
    position = base_position * leverage
    asset_return = prices.pct_change(fill_method=None)
    gross_return = position * asset_return
    turnover = position.diff().abs().fillna(0.0)
    trading_cost = turnover * transaction_cost
    if abs(position.iloc[-1]) > 0:
        trading_cost.iloc[-1] += abs(position.iloc[-1]) * transaction_cost
    active_notional = position.abs()
    roll_haircut = active_notional * annual_roll_haircut / 252.0
    financing = active_notional.gt(0).astype(float) * max(leverage - 1.0, 0.0) * annual_financing / 252.0
    raw_net_return = gross_return - trading_cost - roll_haircut - financing
    net_return = apply_bankruptcy(raw_net_return)
    detail = pd.DataFrame({
        "price": prices, "average_200d": average, "position": position,
        "gross_return": gross_return, "turnover": turnover,
        "trading_cost": trading_cost, "roll_haircut": roll_haircut,
        "financing_cost": financing, "raw_net_return": raw_net_return,
        "net_return": net_return,
    }).dropna(subset=["net_return"])
    detail["equity"] = 10000.0 * (1.0 + detail["net_return"]).cumprod()
    detail["bankrupt"] = detail["equity"].eq(0.0)
    return detail["net_return"], detail, position.loc[detail.index], targets
