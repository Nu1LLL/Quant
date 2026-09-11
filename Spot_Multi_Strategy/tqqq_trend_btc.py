"""Monthly QQQ trend control for TQQQ and alignment with BTC trend."""
import pandas as pd

from cboe_options_benchmark import summarize_returns


def build_tactical_position(qqq_level):
    moving_average = qqq_level.rolling(200, min_periods=200).mean()
    target = (qqq_level > moving_average).astype(float).where(
        moving_average.notna(), 0.0
    )
    month_keys = pd.Series(
        qqq_level.index.year * 100 + qqq_level.index.month,
        index=qqq_level.index,
    )
    month_end = month_keys.ne(month_keys.shift(-1))
    scheduled = target.where(month_end).ffill().fillna(0.0)
    return scheduled.shift(1).fillna(0.0)


def run_etf_sleeve(
    tqqq_level, position, start="2016-09-12", end="2026-09-10",
    transaction_cost=0.0005
):
    returns = tqqq_level.pct_change(fill_method=None).fillna(0.0)
    frame = pd.DataFrame({
        "price": tqqq_level, "asset_return": returns,
        "position": position.reindex(tqqq_level.index).fillna(0.0),
    }).loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")].copy()
    turnover = frame["position"].diff().abs()
    turnover.iloc[0] = abs(frame["position"].iloc[0])
    trading_cost = turnover * transaction_cost
    net_pnl = frame["position"] * frame["asset_return"] - trading_cost
    simulation, yearly, metrics = summarize_returns(net_pnl)
    simulation["price"] = frame["price"].values
    simulation["asset_return"] = frame["asset_return"].values
    simulation["position"] = frame["position"].values
    simulation["turnover"] = turnover.values
    simulation["trading_cost"] = trading_cost.values
    simulation["cost"] = trading_cost.values
    simulation["equity"] = (1.0 + net_pnl).cumprod().values
    simulation["rebalanced"] = (turnover > 0.0).values
    metrics["total_trading_cost"] = float(trading_cost.sum() * 10000.0)
    return simulation, yearly, metrics


def align_with_btc(tactical_simulation, btc_simulation):
    tactical = tactical_simulation.copy()
    tactical["open_time"] = pd.to_datetime(tactical["open_time"], utc=True)
    tactical_equity = tactical.set_index("open_time")["equity"].sort_index()
    btc = btc_simulation.copy()
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc_equity = btc.set_index("open_time")["equity"].sort_index()
    common = tactical_equity.index.intersection(btc_equity.index)
    return pd.DataFrame({
        "TACTICAL_TQQQ": tactical_equity.reindex(common).pct_change(
            fill_method=None
        ).fillna(0.0),
        "BTC_TSMOM30_DD": btc_equity.reindex(common).pct_change(
            fill_method=None
        ).fillna(0.0),
    }, index=common)
