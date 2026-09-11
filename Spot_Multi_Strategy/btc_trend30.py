"""Fixed 30-observation BTC trend and drawdown-state de-risking replication."""
import numpy as np
import pandas as pd

import portfolio_metrics


def build_positions(price, candidate):
    if candidate == "BUY_HOLD":
        target = pd.Series(1.0, index=price.index)
    else:
        momentum = np.log(price / price.shift(30))
        target = (momentum > 0).astype(float).where(momentum.notna(), 0.0)
        if candidate == "TSMOM30_DD":
            drawdown = price / price.cummax() - 1.0
            target = target * pd.Series(
                np.where(drawdown <= -0.15, 0.5, 1.0), index=price.index
            )
        elif candidate != "TSMOM30":
            raise ValueError(f"未知candidate: {candidate}")
    return target.shift(1).fillna(0.0)


def run_backtest(
    price, candidate, leverage=1.0, start="2016-09-12",
    end="2026-09-10", transaction_cost=0.00125,
    annual_financing=0.04, initial_capital=10000.0
):
    base_position = build_positions(price, candidate)
    returns = price.pct_change(fill_method=None).fillna(0.0)
    frame = pd.DataFrame({
        "price": price,
        "asset_return": returns,
        "position": base_position * leverage,
    }).loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")].copy()
    turnover = frame["position"].diff().abs()
    turnover.iloc[0] = abs(frame["position"].iloc[0])
    trading_cost = turnover * transaction_cost
    financing_cost = (
        (frame["position"].abs() - 1.0).clip(lower=0.0)
        * annual_financing / 365.25
    )
    net_pnl = (
        frame["position"] * frame["asset_return"]
        - trading_cost - financing_cost
    )
    simulation = pd.DataFrame({
        "open_time": frame.index,
        "price": frame["price"].values,
        "asset_return": frame["asset_return"].values,
        "position": frame["position"].values,
        "turnover": turnover.values,
        "trading_cost": trading_cost.values,
        "financing_cost": financing_cost.values,
        "cost": (trading_cost + financing_cost).values,
        "net_pnl": net_pnl.values,
        "equity": (1.0 + net_pnl).cumprod().values,
        "rebalanced": (turnover > 0).values,
    })
    metrics = portfolio_metrics.calculate_extended_metrics(
        simulation, initial_capital=initial_capital
    )
    yearly = portfolio_metrics.calendar_year_returns(
        simulation["open_time"], simulation["net_pnl"]
    )
    counts = simulation.assign(
        year=pd.to_datetime(simulation["open_time"], utc=True).dt.year
    ).groupby("year").size()
    complete_years = yearly[counts >= 350]
    rolling_mean = net_pnl.rolling(1095, min_periods=1095).mean()
    rolling_std = net_pnl.rolling(1095, min_periods=1095).std(ddof=0)
    rolling_sharpe = (
        rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(365.25)
    )
    metrics.update({
        "positive_complete_year_ratio": (
            float((complete_years > 0).mean()) if len(complete_years) else 0.0
        ),
        "complete_year_count": int(len(complete_years)),
        "worst_rolling_3y_sharpe": (
            float(rolling_sharpe.min()) if rolling_sharpe.notna().any() else np.nan
        ),
        "total_trading_cost": float(trading_cost.sum() * initial_capital),
        "total_financing_cost": float(financing_cost.sum() * initial_capital),
    })
    return simulation, yearly, metrics
