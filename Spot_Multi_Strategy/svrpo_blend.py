"""Fixed equal-capital blend for a growth sleeve and Cboe SVRPO."""
import numpy as np
import pandas as pd


def align_levels(growth_returns, svrpo_level, start, end):
    growth_returns = growth_returns.sort_index().loc[start:end]
    svrpo_level = svrpo_level.sort_index().loc[start:end]
    common = growth_returns.index.intersection(svrpo_level.index)
    if common.empty or common[0] != start or common[-1] != end:
        raise ValueError("SVRPO does not cover the preregistered ten-year window")
    growth_level = (1.0 + growth_returns.reindex(common)).cumprod()
    return pd.DataFrame({
        "GROWTH_BLEND": growth_level,
        "SVRPO": svrpo_level.reindex(common),
    }, index=common)


def equal_capital_returns(levels, rebalance_every=21, transaction_cost=0.0005):
    asset_returns = levels.pct_change(fill_method=None).fillna(0.0)
    target = np.full(2, 0.5)
    weights = np.zeros(2)
    net_returns = []
    positions = []
    turnovers = []
    for row_number, row in enumerate(asset_returns.to_numpy(dtype=float)):
        if row_number % rebalance_every == 0:
            turnover = float(np.abs(target - weights).sum())
            weights = target.copy()
        else:
            turnover = 0.0
        positions.append(weights.copy())
        gross_return = float(np.dot(weights, row))
        net_returns.append(gross_return - turnover * transaction_cost)
        turnovers.append(turnover)
        denominator = 1.0 + gross_return
        if denominator > 0:
            weights = weights * (1.0 + row) / denominator
    return (
        pd.Series(net_returns, index=levels.index, name="net_return"),
        pd.DataFrame(positions, index=levels.index, columns=levels.columns),
        pd.Series(turnovers, index=levels.index, name="turnover"),
    )
