"""Monthly TQQQ/FMF regime switch driven by the QQQ 200-day trend."""
import pandas as pd

from cboe_options_benchmark import summarize_returns


def build_switch_positions(qqq, tqqq, fmf):
    levels = pd.concat(
        [qqq.rename("QQQ"), tqqq.rename("TQQQ"), fmf.rename("FMF")],
        axis=1, join="inner"
    ).dropna()
    moving_average = levels["QQQ"].rolling(200, min_periods=200).mean()
    valid = moving_average.notna()
    target_tqqq = (levels["QQQ"] > moving_average).astype(float).where(valid)
    targets = pd.DataFrame({
        "TQQQ": target_tqqq, "FMF": (1.0 - target_tqqq).where(valid)
    }, index=levels.index)
    month_keys = pd.Series(
        levels.index.year * 100 + levels.index.month, index=levels.index
    )
    month_end = month_keys.ne(month_keys.shift(-1))
    return targets.where(month_end, axis=0).ffill().fillna(0.0).shift(1).fillna(0.0)


def run_switch_sleeve(
    qqq, tqqq, fmf, start="2016-09-12", end="2026-09-10",
    transaction_cost=0.0005
):
    levels = pd.concat(
        [qqq.rename("QQQ"), tqqq.rename("TQQQ"), fmf.rename("FMF")],
        axis=1, join="inner"
    ).dropna()
    positions = build_switch_positions(qqq, tqqq, fmf)
    returns = levels[["TQQQ", "FMF"]].pct_change(fill_method=None).fillna(0.0)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    positions = positions.loc[start_ts:end_ts].copy()
    returns = returns.reindex(positions.index)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    trading_cost = turnover * transaction_cost
    net_pnl = (positions * returns).sum(axis=1) - trading_cost
    simulation, yearly, metrics = summarize_returns(net_pnl)
    simulation["tqqq_position"] = positions["TQQQ"].values
    simulation["fmf_position"] = positions["FMF"].values
    simulation["turnover"] = turnover.values
    simulation["trading_cost"] = trading_cost.values
    simulation["cost"] = trading_cost.values
    simulation["equity"] = (1.0 + net_pnl).cumprod().values
    simulation["rebalanced"] = (turnover > 0.0).values
    metrics["total_trading_cost"] = float(trading_cost.sum() * 10000.0)
    return simulation, positions, yearly, metrics
