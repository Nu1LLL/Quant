"""Causal volatility target and drawdown overlay for the fixed PPUT/BTC blend."""
import numpy as np
import pandas as pd

from cboe_options_benchmark import summarize_returns


def drawdown_scalar(drawdown, soft=0.08, hard=0.20, minimum=0.30):
    depth = -drawdown
    if depth <= soft:
        return 1.0
    if depth >= hard:
        return minimum
    return 1.0 - (depth - soft) / (hard - soft) * (1.0 - minimum)


def run_overlay(
    base_returns, vol_window=63, vol_target=0.25, exposure_cap=3.0,
    no_trade_band=0.05, transaction_cost=0.0005,
    annual_financing=0.04
):
    trailing_vol = (
        base_returns.rolling(vol_window, min_periods=vol_window)
        .std(ddof=0).shift(1) * np.sqrt(252.0)
    )
    raw_target = (
        vol_target / trailing_vol.replace(0, np.nan)
    ).clip(upper=exposure_cap).fillna(0.0)

    n = len(base_returns)
    positions = np.zeros(n)
    dd_scalars = np.ones(n)
    turnovers = np.zeros(n)
    financing = np.zeros(n)
    costs = np.zeros(n)
    net_returns = np.zeros(n)
    equity_path = np.zeros(n)
    equity = 1.0
    running_max = 1.0
    current_position = 0.0

    for i, base_return in enumerate(base_returns.to_numpy(dtype=float)):
        drawdown = equity / running_max - 1.0
        scalar = drawdown_scalar(drawdown)
        target = float(raw_target.iloc[i]) * scalar
        dd_scalars[i] = scalar
        if abs(target - current_position) >= no_trade_band:
            turnover = abs(target - current_position)
            current_position = target
        else:
            turnover = 0.0
        finance = (
            max(current_position - 1.0, 0.0) * annual_financing / 252.0
        )
        cost = turnover * transaction_cost + finance
        net_return = current_position * base_return - cost
        equity *= 1.0 + net_return
        running_max = max(running_max, equity)

        positions[i] = current_position
        turnovers[i] = turnover
        financing[i] = finance
        costs[i] = cost
        net_returns[i] = net_return
        equity_path[i] = equity

    simulation = pd.DataFrame({
        "open_time": base_returns.index,
        "base_return": base_returns.values,
        "raw_vol_target": raw_target.values,
        "drawdown_scalar": dd_scalars,
        "position": positions,
        "turnover": turnovers,
        "financing_cost": financing,
        "cost": costs,
        "net_pnl": net_returns,
        "equity": equity_path,
        "rebalanced": turnovers > 0,
    })
    returns = pd.Series(net_returns, index=base_returns.index)
    _, yearly, metrics = summarize_returns(returns)
    metrics.update({
        "average_position": float(np.mean(positions)),
        "max_position": float(np.max(positions)),
        "total_overlay_cost": float(np.sum(costs) * 10000.0),
    })
    return simulation, yearly, metrics
