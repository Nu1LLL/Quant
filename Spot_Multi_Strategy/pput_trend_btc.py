"""Faber-style PPUT trend filter blended with the fixed BTC trend sleeve."""
import numpy as np
import pandas as pd

import blend_risk_overlay
from cboe_options_benchmark import summarize_returns


def build_pput_trend_returns(level, transaction_cost=0.0005):
    daily_return = level.pct_change(fill_method=None).fillna(0.0)
    moving_average = level.rolling(200, min_periods=200).mean()
    target = (level > moving_average).astype(float).where(
        moving_average.notna(), 0.0
    )
    periods = pd.Series(
        level.index.year * 100 + level.index.month, index=level.index
    )
    # Series.shift is positional, so compare each YYYYMM key with the next row.
    month_end = periods.ne(periods.shift(-1))
    scheduled = target.where(month_end).ffill().fillna(0.0)
    position = scheduled.shift(1).fillna(0.0)
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    net_return = position * daily_return - turnover * transaction_cost
    return net_return, position, turnover


def build_blend_positions(
    component_returns, vol_window=60, rebalance_every=21, vol_floor=0.01
):
    annualized_vol = component_returns.rolling(
        vol_window, min_periods=vol_window
    ).std(ddof=0) * np.sqrt(252.0)
    inverse_vol = 1.0 / annualized_vol.clip(lower=vol_floor)
    targets = inverse_vol.div(inverse_vol.sum(axis=1), axis=0)
    mask = pd.Series(False, index=component_returns.index)
    mask.iloc[vol_window::rebalance_every] = True
    scheduled = targets.where(mask, axis=0).ffill().fillna(0.0)
    return scheduled.shift(1).fillna(0.0)


def run_base_blend(component_returns, transaction_cost=0.0005):
    positions = build_blend_positions(component_returns)
    gross_return = (positions * component_returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    return gross_return - turnover * transaction_cost, positions, turnover


def run_static_scenario(component_returns, leverage, annual_financing=0.04):
    base_return, positions, turnover = run_base_blend(component_returns)
    active = positions.abs().sum(axis=1)
    financing = (
        (active * leverage - 1.0).clip(lower=0.0)
        * annual_financing / 252.0
    )
    returns = base_return * leverage - financing
    simulation, yearly, metrics = summarize_returns(returns)
    simulation["gross_exposure"] = (active * leverage).values
    simulation["turnover"] = (turnover * leverage).values
    simulation["financing_cost"] = financing.values
    simulation["equity"] = (1.0 + returns).cumprod().values
    return simulation, positions, yearly, metrics


def run_overlay_scenario(component_returns):
    base_return, positions, turnover = run_base_blend(component_returns)
    simulation, yearly, metrics = blend_risk_overlay.run_overlay(base_return)
    return simulation, positions, yearly, metrics
