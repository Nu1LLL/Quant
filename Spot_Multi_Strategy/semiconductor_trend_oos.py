"""Causal monthly SOXX trend switch between long SOXL and short SOXX."""
import numpy as np
import pandas as pd


SYMBOLS = ("SOXX", "SOXL")
LATEST_ACCEPTABLE_START = pd.Timestamp("2010-04-30", tz="UTC")


def validate_prices(prices):
    frame = pd.DataFrame(prices).copy().sort_index()
    if set(frame.columns) != set(SYMBOLS):
        raise ValueError(f"Expected exactly {SYMBOLS}")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame.loc[:, list(SYMBOLS)].dropna()
    values = frame.to_numpy(dtype=float)
    if frame.empty or frame.index.has_duplicates or not np.isfinite(values).all():
        raise ValueError("Semiconductor panel is empty, duplicated or non-finite")
    if (values <= 0).any():
        raise ValueError("Semiconductor prices must be positive")
    gaps = frame.index.to_series().diff().dropna().dt.days
    if not gaps.empty and int(gaps.max()) > 10:
        raise ValueError("Semiconductor common panel has a gap longer than ten days")
    return frame


def causal_weights(prices, window=200):
    prices = validate_prices(prices)
    average = prices["SOXX"].rolling(window, min_periods=window).mean()
    risk_on = prices["SOXX"].gt(average).where(average.notna())
    targets = pd.DataFrame({
        "SOXX": risk_on.map({True: 0.0, False: -1.0}),
        "SOXL": risk_on.map({True: 1.0, False: 0.0}),
    }, index=prices.index)
    month = pd.Series(prices.index.year * 100 + prices.index.month, index=prices.index)
    month_end = month.ne(month.shift(-1))
    targets = targets.where(month_end, axis=0).ffill().fillna(0.0)
    return targets.shift(2).fillna(0.0), average, targets


def apply_bankruptcy(raw_returns):
    reported = pd.Series(raw_returns).copy()
    bankrupt = reported.le(-1.0)
    if bankrupt.any():
        first = bankrupt[bankrupt].index[0]
        reported.loc[first] = -1.0
        reported.loc[reported.index > first] = 0.0
    return reported


def run_scenario(
    prices, multiplier, leg_cost=0.0005, annual_borrow=0.005,
    annual_financing=0.04,
):
    prices = validate_prices(prices)
    base_weights, average, targets = causal_weights(prices)
    weights = base_weights * multiplier
    asset_returns = prices.pct_change(fill_method=None)
    gross_return = (weights * asset_returns).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    maintenance_turnover = (weights * asset_returns).abs().sum(axis=1, min_count=1)
    trading_cost = (turnover + maintenance_turnover.fillna(0.0)) * leg_cost
    if weights.iloc[-1].abs().sum() > 0:
        trading_cost.iloc[-1] += weights.iloc[-1].abs().sum() * leg_cost
    short_notional = weights["SOXX"].clip(upper=0).abs()
    short_borrow = short_notional * annual_borrow / 252.0
    active = weights.abs().sum(axis=1).gt(0).astype(float)
    financing = active * max(multiplier - 1.0, 0.0) * annual_financing / 252.0
    raw_net_return = gross_return - trading_cost - short_borrow - financing
    net_return = apply_bankruptcy(raw_net_return)
    detail = pd.DataFrame({
        "soxx_price": prices["SOXX"], "soxx_200d_average": average,
        "weight_SOXX": weights["SOXX"], "weight_SOXL": weights["SOXL"],
        "gross_return": gross_return, "turnover": turnover,
        "maintenance_turnover": maintenance_turnover, "trading_cost": trading_cost,
        "short_borrow": short_borrow, "financing_cost": financing,
        "raw_net_return": raw_net_return, "net_return": net_return,
    }).dropna(subset=["net_return"])
    detail["equity"] = 10000.0 * (1.0 + detail["net_return"]).cumprod()
    detail["bankrupt"] = detail["equity"].eq(0.0)
    return detail["net_return"], detail, weights.loc[detail.index], targets
