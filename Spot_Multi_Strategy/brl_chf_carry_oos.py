"""Dollar-neutral BZF versus FXF currency-carry proxy."""
import numpy as np
import pandas as pd


SYMBOLS = ("BZF", "FXF")
LATEST_ACCEPTABLE_START = pd.Timestamp("2009-01-31", tz="UTC")


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
        raise ValueError("BRL/CHF panel is empty, duplicated or non-finite")
    if (values <= 0).any():
        raise ValueError("BRL/CHF prices must be positive")
    gaps = frame.index.to_series().diff().dropna().dt.days
    if not gaps.empty and int(gaps.max()) > 10:
        raise ValueError("BRL/CHF common panel has a gap longer than ten days")
    return frame


def run_scenario(
    prices, leverage, leg_cost=0.001, annual_borrow=0.005,
    annual_financing=0.04,
):
    prices = validate_prices(prices)
    target = pd.DataFrame({"BZF": 0.5, "FXF": -0.5}, index=prices.index)
    weights = target.shift(2).fillna(0.0) * leverage
    asset_returns = prices.pct_change(fill_method=None)
    gross_return = (weights * asset_returns).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    maintenance_turnover = (weights * asset_returns).abs().sum(axis=1, min_count=1)
    trading_cost = (turnover + maintenance_turnover.fillna(0.0)) * leg_cost
    if weights.iloc[-1].abs().sum() > 0:
        trading_cost.iloc[-1] += weights.iloc[-1].abs().sum() * leg_cost
    short_borrow = weights["FXF"].clip(upper=0.0).abs() * annual_borrow / 252.0
    active = weights.abs().sum(axis=1).gt(0).astype(float)
    financing = active * max(leverage - 1.0, 0.0) * annual_financing / 252.0
    net_return = gross_return - trading_cost - short_borrow - financing
    detail = pd.DataFrame({
        "weight_BZF": weights["BZF"], "weight_FXF": weights["FXF"],
        "gross_return": gross_return, "turnover": turnover,
        "maintenance_turnover": maintenance_turnover, "trading_cost": trading_cost,
        "short_borrow": short_borrow, "financing_cost": financing,
        "net_return": net_return,
    }).dropna(subset=["net_return"])
    detail["equity"] = 10000.0 * (1.0 + detail["net_return"]).cumprod()
    return detail["net_return"], detail, weights.loc[detail.index]
