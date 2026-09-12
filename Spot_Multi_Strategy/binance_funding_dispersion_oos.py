"""Binance public-data loader and lagged cross-sectional funding strategy."""
from pathlib import Path

import numpy as np
import pandas as pd

from binance_basis_oos import download_funding, download_klines


EIGHT_HOURS = pd.Timedelta(hours=8)


def load_symbol(symbol, start, end, cache_dir, refresh=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol}_funding_dispersion_inputs.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["funding_time"])
    funding = download_funding(symbol, start, end)
    mark = download_klines(symbol, start, end, mark=True).rename(
        columns={"price": "mark_price"}
    )
    frame = funding.merge(mark, on="funding_time", how="inner")
    frame.to_csv(path, index=False)
    return frame


def build_common_panel(frames):
    pieces = []
    for symbol in sorted(frames):
        frame = frames[symbol].copy().set_index("funding_time").sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate Binance timestamp for {symbol}")
        if (frame["mark_price"] <= 0).any():
            raise ValueError(f"Nonpositive Binance mark price for {symbol}")
        pieces.append(frame[["funding_rate", "mark_price"]].rename(columns={
            "funding_rate": f"{symbol}__funding",
            "mark_price": f"{symbol}__mark",
        }))
    panel = pd.concat(pieces, axis=1, join="inner").dropna().sort_index()
    if panel.index.has_duplicates:
        raise ValueError("Duplicate common Binance timestamps")
    return panel


def coverage_audit(panel):
    if panel.empty:
        return {
            "observations": 0, "start": None, "end": None,
            "coverage_ratio": 0.0, "max_gap_hours": np.inf,
            "scheduled_hours_valid": False, "duplicate_count": 0,
            "span_years": 0.0, "passed": False,
        }
    index = pd.DatetimeIndex(panel.index).tz_convert("UTC")
    expected = pd.date_range(index[0], index[-1], freq="8h", tz="UTC")
    gaps = index.to_series().diff().dropna().dt.total_seconds().div(3600)
    audit = {
        "observations": int(len(index)),
        "start": index[0],
        "end": index[-1],
        "coverage_ratio": float(len(index.intersection(expected)) / len(expected)),
        "max_gap_hours": float(gaps.max()) if not gaps.empty else 0.0,
        "scheduled_hours_valid": bool(index.hour.isin([0, 8, 16]).all()),
        "duplicate_count": int(index.duplicated().sum()),
        "span_years": float((index[-1] - index[0]).days / 365.25),
    }
    audit["passed"] = bool(
        index[0] <= pd.Timestamp("2020-03-31", tz="UTC")
        and audit["span_years"] >= 6.0
        and audit["coverage_ratio"] >= 0.99
        and audit["max_gap_hours"] <= 16.0
        and audit["scheduled_hours_valid"]
        and audit["duplicate_count"] == 0
    )
    return audit


def lagged_rank_weights(panel):
    symbols = sorted(
        column.split("__")[0] for column in panel if column.endswith("__funding")
    )
    rates = panel[[f"{symbol}__funding" for symbol in symbols]].copy()
    rates.columns = symbols
    targets = pd.DataFrame(0.0, index=panel.index, columns=symbols)
    for timestamp, row in rates.iterrows():
        ordered = sorted(symbols, key=lambda symbol: (float(row[symbol]), symbol))
        targets.at[timestamp, ordered[0]] = 0.5
        targets.at[timestamp, ordered[-1]] = -0.5
    return targets.shift(1), rates.max(axis=1).sub(rates.min(axis=1)).shift(1)


def run_notional(panel, leverage, leg_cost=0.0005, annual_financing=0.04):
    positions, lagged_spread = lagged_rank_weights(panel)
    symbols = list(positions.columns)
    marks = panel[[f"{symbol}__mark" for symbol in symbols]].copy()
    marks.columns = symbols
    funding = panel[[f"{symbol}__funding" for symbol in symbols]].copy()
    funding.columns = symbols
    price_returns = marks.pct_change(fill_method=None)
    weights = positions * leverage
    gross_price = (weights * price_returns).sum(axis=1, min_count=1)
    funding_pnl = (-weights * funding).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1)
    first_valid = weights.dropna(how="all").index[0]
    turnover.loc[first_valid] = weights.loc[first_valid].abs().sum()
    last_valid = weights.dropna(how="all").index[-1]
    trading_cost = turnover * leg_cost
    trading_cost.loc[last_valid] += weights.loc[last_valid].abs().sum() * leg_cost
    financing = max(leverage - 1.0, 0.0) * annual_financing / 1095.0
    interval_return = gross_price + funding_pnl - trading_cost - financing
    detail = pd.DataFrame({
        "gross_price_return": gross_price,
        "funding_pnl": funding_pnl,
        "turnover": turnover,
        "trading_cost": trading_cost,
        "financing_cost": financing,
        "lagged_funding_spread": lagged_spread,
        "net_return": interval_return,
    }).dropna(subset=["net_return"])
    daily = detail["net_return"].groupby(detail.index.normalize()).apply(
        lambda values: float((1.0 + values).prod() - 1.0)
    )
    daily.name = "net_return"
    return daily, detail, weights.loc[detail.index]


def diagnostic_labels(detail):
    daily_spread = detail["lagged_funding_spread"].groupby(
        detail.index.normalize()
    ).mean()
    median = float(daily_spread.median())
    labels = pd.DataFrame({
        "funding_spread_regime": np.where(
            daily_spread >= median, "spread_high", "spread_low"
        ),
        "calendar_year": daily_spread.index.year.astype(str),
    }, index=daily_spread.index)
    return labels, median
