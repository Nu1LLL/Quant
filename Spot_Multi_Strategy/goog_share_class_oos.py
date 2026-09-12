"""Causal mean reversion in the GOOG/GOOGL dual-share-class spread."""
import numpy as np
import pandas as pd


SYMBOLS = ("GOOG", "GOOGL")
LATEST_ACCEPTABLE_START = pd.Timestamp("2014-04-30", tz="UTC")


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
        raise ValueError("Alphabet price panel is empty, duplicated or non-finite")
    if (values <= 0).any():
        raise ValueError("Alphabet prices must be positive")
    gaps = frame.index.to_series().diff().dropna().dt.days
    if not gaps.empty and int(gaps.max()) > 10:
        raise ValueError("Alphabet common panel has a gap longer than ten days")
    return frame


def causal_state(prices, window=252, entry_z=1.0):
    prices = validate_prices(prices)
    spread = np.log(prices["GOOG"] / prices["GOOGL"])
    prior_mean = spread.shift(1).rolling(window, min_periods=window).mean()
    prior_std = spread.shift(1).rolling(window, min_periods=window).std(ddof=0)
    zscore = spread.sub(prior_mean).div(prior_std.replace(0.0, np.nan))
    state = pd.Series(0.0, index=prices.index, name="target_state")
    current = 0.0
    for timestamp, zvalue in zscore.items():
        if np.isfinite(zvalue):
            if current == 0.0:
                if zvalue >= entry_z:
                    current = -1.0
                elif zvalue <= -entry_z:
                    current = 1.0
            elif current > 0.0 and zvalue >= 0.0:
                current = 0.0
            elif current < 0.0 and zvalue <= 0.0:
                current = 0.0
        state.at[timestamp] = current
    target = pd.DataFrame({
        "GOOG": 0.5 * state,
        "GOOGL": -0.5 * state,
    }, index=prices.index)
    executed = target.shift(2).fillna(0.0)
    diagnostics = pd.DataFrame({
        "spread": spread,
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "zscore": zscore,
        "target_state": state,
        "executed_state": executed["GOOG"] * 2.0,
    })
    return executed, diagnostics


def run_scenario(
    prices, leverage, leg_cost=0.0002, annual_borrow=0.01,
    annual_financing=0.04,
):
    prices = validate_prices(prices)
    base_weights, signals = causal_state(prices)
    asset_returns = prices.pct_change(fill_method=None)
    weights = base_weights * leverage
    gross_return = (weights * asset_returns).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    maintenance_turnover = (weights * asset_returns).abs().sum(axis=1, min_count=1)
    trading_cost = (turnover + maintenance_turnover.fillna(0.0)) * leg_cost
    if weights.iloc[-1].abs().sum() > 0:
        trading_cost.iloc[-1] += weights.iloc[-1].abs().sum() * leg_cost
    short_notional = weights.clip(upper=0.0).abs().sum(axis=1)
    active_gross = weights.abs().sum(axis=1)
    short_borrow = short_notional * annual_borrow / 252.0
    financing = (
        active_gross.gt(0).astype(float)
        * max(leverage - 1.0, 0.0) * annual_financing / 252.0
    )
    net_return = gross_return - trading_cost - short_borrow - financing
    detail = pd.concat([
        signals,
        weights.add_prefix("weight_"),
        pd.DataFrame({
            "gross_return": gross_return,
            "turnover": turnover,
            "maintenance_turnover": maintenance_turnover,
            "trading_cost": trading_cost,
            "short_borrow": short_borrow,
            "financing_cost": financing,
            "net_return": net_return,
        }, index=prices.index),
    ], axis=1).dropna(subset=["net_return"])
    detail["equity"] = 10000.0 * (1.0 + detail["net_return"]).cumprod()
    return detail["net_return"], detail, weights.loc[detail.index]
