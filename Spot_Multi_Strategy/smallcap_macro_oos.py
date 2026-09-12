"""Monthly IWM trend switch between TNA and a fixed TBT/UGL defensive pair."""
import pandas as pd

from cboe_options_benchmark import summarize_returns


def build_positions(iwm, tna, tbt, ugl):
    levels = pd.concat(
        [iwm.rename("IWM"), tna.rename("TNA"), tbt.rename("TBT"), ugl.rename("UGL")],
        axis=1, join="inner"
    ).dropna()
    moving_average = levels["IWM"].rolling(200, min_periods=200).mean()
    valid = moving_average.notna()
    risk_on = (levels["IWM"] > moving_average).astype(float).where(valid)
    targets = pd.DataFrame({
        "TNA": risk_on,
        "TBT": ((1.0-risk_on)*0.5).where(valid),
        "UGL": ((1.0-risk_on)*0.5).where(valid),
    }, index=levels.index)
    month_key = pd.Series(levels.index.year*100 + levels.index.month, index=levels.index)
    month_end = month_key.ne(month_key.shift(-1))
    return targets.where(month_end, axis=0).ffill().fillna(0.0).shift(1).fillna(0.0)


def run_strategy(iwm, tna, tbt, ugl, start, end, transaction_cost=0.0005):
    levels = pd.concat(
        [iwm.rename("IWM"), tna.rename("TNA"), tbt.rename("TBT"), ugl.rename("UGL")],
        axis=1, join="inner"
    ).dropna()
    positions = build_positions(iwm, tna, tbt, ugl)
    asset_returns = levels[["TNA", "TBT", "UGL"]].pct_change(fill_method=None).fillna(0.0)
    positions = positions.loc[start:end]
    asset_returns = asset_returns.reindex(positions.index)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    cost = turnover * transaction_cost
    net_return = (positions*asset_returns).sum(axis=1) - cost
    simulation, yearly, metrics = summarize_returns(net_return)
    simulation["equity"] = (1+net_return).cumprod().values
    simulation["turnover"] = turnover.values
    simulation["cost"] = cost.values
    simulation["risk_on"] = positions["TNA"].values
    simulation["rebalanced"] = (turnover > 0).values
    return simulation, positions, yearly, metrics
