"""Causal fixed-weight drifting portfolio with scheduled rebalancing."""
import numpy as np
import pandas as pd


def run_fixed_weight(levels, target_weights, rebalance_every=21, cost=0.0005):
    target = pd.Series(target_weights, dtype=float).reindex(levels.columns)
    if target.isna().any() or (target < 0).any() or not np.isclose(target.sum(), 1.0):
        raise ValueError("Target weights must be nonnegative and sum to one")
    asset_returns = levels.pct_change(fill_method=None).fillna(0.0)
    weights = np.zeros(len(target))
    returns, positions, turnovers = [], [], []
    for row_number, row in enumerate(asset_returns.to_numpy(dtype=float)):
        if row_number % rebalance_every == 0:
            turnover = float(np.abs(target.to_numpy() - weights).sum())
            weights = target.to_numpy().copy()
        else:
            turnover = 0.0
        positions.append(weights.copy())
        gross = float(np.dot(weights, row))
        returns.append(gross - turnover * cost)
        turnovers.append(turnover)
        if 1.0 + gross > 0:
            weights = weights * (1.0 + row) / (1.0 + gross)
    return (
        pd.Series(returns, index=levels.index, name="net_return"),
        pd.DataFrame(positions, index=levels.index, columns=levels.columns),
        pd.Series(turnovers, index=levels.index, name="turnover"),
    )
