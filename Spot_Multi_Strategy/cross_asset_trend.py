"""Pre-registered multi-horizon trend strategy for liquid cross-asset ETFs."""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import portfolio_metrics

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0"
DEFAULT_SYMBOLS = [
    "SPY", "EFA", "EEM", "VNQ",
    "IEF", "TLT", "LQD", "HYG",
    "GLD", "DBC", "USO",
    "UUP", "FXE", "FXY",
]


def download_adjusted_close(symbol, start, end, timeout=20, session=None):
    """Download Yahoo adjusted close for a fixed half-open date interval."""
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = pd.Timestamp(end, tz="UTC")
    request_session = session or requests.Session()
    response = request_session.get(
        YAHOO_CHART_URL.format(symbol=symbol),
        params={
            "period1": int(start_time.timestamp()),
            "period2": int(end_time.timestamp()),
            "interval": "1d",
            "events": "div,splits",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"Yahoo Finance没有返回{symbol}的数据")
    result = result[0]
    adjusted = result.get("indicators", {}).get("adjclose")
    if not adjusted or "adjclose" not in adjusted[0]:
        raise ValueError(f"Yahoo Finance没有返回{symbol}的调整收盘价")
    series = pd.Series(
        adjusted[0]["adjclose"],
        index=pd.to_datetime(result["timestamp"], unit="s", utc=True),
        name=symbol,
        dtype=float,
    )
    return series.dropna().sort_index()


def load_adjusted_close(
    symbol, start, end, cache_folder="cross_asset_data_cache", refresh=False
):
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)
    file_path = cache_folder / f"{symbol}_{start}_{end}.csv"
    if file_path.exists() and not refresh:
        frame = pd.read_csv(file_path, parse_dates=["open_time"])
        return frame.set_index("open_time")["adjusted_close"].rename(symbol)
    series = download_adjusted_close(symbol, start, end)
    series.rename("adjusted_close").rename_axis("open_time").to_csv(file_path)
    return series


def load_price_matrix(symbols, start, end, cache_folder, refresh=False):
    series = [
        load_adjusted_close(
            symbol, start, end, cache_folder=cache_folder, refresh=refresh
        )
        for symbol in symbols
    ]
    return pd.concat(series, axis=1, join="inner").dropna()


def _cap_gross(weights, gross_cap):
    gross = weights.abs().sum(axis=1)
    scalar = (gross_cap / gross.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    return weights.mul(scalar, axis=0)


def build_trend_positions(
    prices, leverage=1.0, lookbacks=(21, 63, 252), vol_window=60,
    base_vol_target=0.10, rebalance_every=21, gross_cap=4.0
):
    """Return positions used for each day's close-to-close return.

    Signals and volatility observed at t are scheduled monthly and shifted once,
    so they first affect the return from t to t+1.
    """
    returns = prices.pct_change(fill_method=None)
    components = [
        np.sign(prices.pct_change(periods=lookback, fill_method=None))
        for lookback in lookbacks
    ]
    signal = sum(components) / len(components)
    annualized_vol = returns.rolling(
        vol_window, min_periods=vol_window
    ).std(ddof=0) * np.sqrt(252.0)
    risk_budget = base_vol_target / np.sqrt(prices.shape[1])
    targets = signal * risk_budget / annualized_vol.replace(0, np.nan)

    first_ready = max(max(lookbacks), vol_window)
    rebalance_mask = pd.Series(False, index=prices.index)
    rebalance_mask.iloc[first_ready::rebalance_every] = True
    scheduled = targets.where(rebalance_mask, axis=0).ffill().fillna(0.0)
    scheduled = _cap_gross(scheduled * leverage, gross_cap)
    return scheduled.shift(1).fillna(0.0)


def run_backtest(
    prices, leverage=1.0, transaction_cost=0.0007,
    annual_short_borrow=0.005, annual_financing=0.04,
    initial_capital=10000.0
):
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    positions = build_trend_positions(prices, leverage=leverage)
    gross_return = (positions * returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    gross_exposure = positions.abs().sum(axis=1)
    short_exposure = positions.clip(upper=0.0).abs().sum(axis=1)
    trading_cost = turnover * transaction_cost
    borrow_cost = short_exposure * annual_short_borrow / 252.0
    financing_cost = (
        (gross_exposure - 1.0).clip(lower=0.0) * annual_financing / 252.0
    )
    total_cost = trading_cost + borrow_cost + financing_cost
    net_pnl = gross_return - total_cost
    equity = (1.0 + net_pnl).cumprod()
    simulation = pd.DataFrame({
        "open_time": prices.index,
        "net_pnl": net_pnl.values,
        "equity": equity.values,
        "position": gross_exposure.values,
        "turnover": turnover.values,
        "cost": total_cost.values,
        "trading_cost": trading_cost.values,
        "borrow_cost": borrow_cost.values,
        "financing_cost": financing_cost.values,
        "rebalanced": (turnover > 0).values,
    })
    metrics = portfolio_metrics.calculate_extended_metrics(
        simulation, initial_capital=initial_capital
    )
    yearly = portfolio_metrics.calendar_year_returns(
        simulation["open_time"], simulation["net_pnl"]
    )
    year_counts = simulation.assign(
        year=pd.to_datetime(simulation["open_time"], utc=True).dt.year
    ).groupby("year").size()
    complete_years = yearly[year_counts >= 240]
    rolling_mean = net_pnl.rolling(756, min_periods=756).mean()
    rolling_std = net_pnl.rolling(756, min_periods=756).std(ddof=0)
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)
    metrics.update({
        "positive_complete_year_ratio": (
            float((complete_years > 0).mean()) if len(complete_years) else 0.0
        ),
        "complete_year_count": int(len(complete_years)),
        "worst_rolling_3y_sharpe": (
            float(rolling_sharpe.min()) if rolling_sharpe.notna().any() else np.nan
        ),
        "total_trading_cost": float(trading_cost.sum() * initial_capital),
        "total_borrow_cost": float(borrow_cost.sum() * initial_capital),
        "total_financing_cost": float(financing_cost.sum() * initial_capital),
    })
    return simulation, positions, yearly, metrics
