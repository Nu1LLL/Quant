"""Cboe official options-strategy benchmark feasibility study."""
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import portfolio_metrics

SYMBOLS = ("PUT", "BXM", "BXMD", "PPUT", "CMBO")
URL_TEMPLATE = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/"
    "{symbol}_History.csv"
)


def download_index(symbol, timeout=20, session=None):
    request_session = session or requests.Session()
    response = request_session.get(URL_TEMPLATE.format(symbol=symbol), timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if "DATE" not in frame or symbol not in frame:
        raise ValueError(f"Cboe {symbol} CSV列格式不符合预期")
    series = pd.Series(
        pd.to_numeric(frame[symbol], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame["DATE"], format="%m/%d/%Y", utc=True),
        name=symbol,
    )
    return series.dropna().sort_index()


def load_index(symbol, cache_folder, refresh=False):
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)
    file_path = cache_folder / f"{symbol}_History.csv"
    if file_path.exists() and not refresh:
        frame = pd.read_csv(file_path, parse_dates=["open_time"])
        return frame.set_index("open_time")["level"].rename(symbol)
    series = download_index(symbol)
    series.rename("level").rename_axis("open_time").to_csv(file_path)
    return series


def load_index_matrix(cache_folder, refresh=False):
    series = [
        load_index(symbol, cache_folder, refresh=refresh) for symbol in SYMBOLS
    ]
    return pd.concat(series, axis=1, join="inner").dropna()


def monthly_equal_weight_returns(levels, rebalance_every=21, cost=0.0005):
    """Simulate five drifting sleeves, reset to equal weight every 21 rows."""
    asset_returns = levels.pct_change(fill_method=None).fillna(0.0)
    n_assets = asset_returns.shape[1]
    target = np.full(n_assets, 1.0 / n_assets)
    weights = target.copy()
    results = []
    turnovers = []
    for row_number, row in enumerate(asset_returns.to_numpy(dtype=float)):
        if row_number % rebalance_every == 0:
            turnover = float(np.abs(target - weights).sum())
            if row_number == 0:
                turnover = 1.0
            weights = target.copy()
        else:
            turnover = 0.0
        gross_return = float(np.dot(weights, row))
        net_return = gross_return - turnover * cost
        results.append(net_return)
        turnovers.append(turnover)
        denominator = 1.0 + gross_return
        if denominator > 0:
            weights = weights * (1.0 + row) / denominator
    return (
        pd.Series(results, index=levels.index, name="OPTIONS_EQUAL"),
        pd.Series(turnovers, index=levels.index, name="turnover"),
    )


def apply_leverage(base_returns, leverage, annual_financing=0.04):
    financing = max(leverage - 1.0, 0.0) * annual_financing / 252.0
    return base_returns * leverage - financing


def summarize_returns(returns, initial_capital=10000.0):
    simulation = pd.DataFrame({
        "open_time": returns.index,
        "net_pnl": returns.values,
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
    complete_years = yearly[counts >= 240]
    rolling_mean = returns.rolling(756, min_periods=756).mean()
    rolling_std = returns.rolling(756, min_periods=756).std(ddof=0)
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)
    metrics.update({
        "positive_complete_year_ratio": (
            float((complete_years > 0).mean()) if len(complete_years) else 0.0
        ),
        "complete_year_count": int(len(complete_years)),
        "worst_rolling_3y_sharpe": (
            float(rolling_sharpe.min()) if rolling_sharpe.notna().any() else np.nan
        ),
    })
    return simulation, yearly, metrics
