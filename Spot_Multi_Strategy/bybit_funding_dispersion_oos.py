"""Bybit public-data loader and lagged cross-sectional funding strategy."""
from pathlib import Path

import numpy as np
import pandas as pd
import requests


FUNDING_URL = "https://api.bybit.com/v5/market/funding/history"
MARK_URL = "https://api.bybit.com/v5/market/mark-price-kline"
EIGHT_HOURS_MS = 8 * 60 * 60 * 1000
FOUR_HOURS_MS = 4 * 60 * 60 * 1000


def _milliseconds(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def _payload(response):
    response.raise_for_status()
    payload = response.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error {payload.get('retCode')}: {payload.get('retMsg')}")
    return payload["result"].get("list", [])


def download_funding(symbol, start, end, session=None):
    client = session or requests.Session()
    start_ms, cursor = _milliseconds(start), _milliseconds(end)
    records = []
    while cursor >= start_ms:
        response = client.get(FUNDING_URL, params={
            "category": "linear", "symbol": symbol,
            "startTime": start_ms, "endTime": cursor, "limit": 200,
        }, timeout=30)
        page = _payload(response)
        if not page:
            break
        records.extend(page)
        earliest = min(int(row["fundingRateTimestamp"]) for row in page)
        if earliest <= start_ms or len(page) < 200:
            break
        next_cursor = earliest - 1
        if next_cursor >= cursor:
            raise RuntimeError("Bybit funding pagination did not advance")
        cursor = next_cursor
    if not records:
        raise ValueError(f"No Bybit funding history for {symbol}")
    frame = pd.DataFrame(records)
    raw_time = pd.to_datetime(
        pd.to_numeric(frame["fundingRateTimestamp"], errors="raise"),
        unit="ms", utc=True,
    )
    scheduled = raw_time.dt.round("8h")
    close_to_schedule = (raw_time - scheduled).abs() <= pd.Timedelta(minutes=5)
    result = pd.DataFrame({
        "funding_time": scheduled[close_to_schedule],
        "funding_rate": pd.to_numeric(frame.loc[close_to_schedule, "fundingRate"], errors="raise"),
    })
    return result.dropna().drop_duplicates("funding_time", keep="last").sort_values("funding_time")


def download_mark_klines(symbol, start, end, session=None):
    client = session or requests.Session()
    start_ms, cursor = _milliseconds(start), _milliseconds(end)
    records = []
    while cursor >= start_ms:
        response = client.get(MARK_URL, params={
            "category": "linear", "symbol": symbol, "interval": "240",
            "start": start_ms - FOUR_HOURS_MS, "end": cursor,
            "limit": 1000,
        }, timeout=30)
        page = _payload(response)
        if not page:
            break
        records.extend(page)
        earliest = min(int(row[0]) for row in page)
        if earliest <= start_ms - FOUR_HOURS_MS or len(page) < 1000:
            break
        next_cursor = earliest - 1
        if next_cursor >= cursor:
            raise RuntimeError("Bybit mark-kline pagination did not advance")
        cursor = next_cursor
    if not records:
        raise ValueError(f"No Bybit mark-price klines for {symbol}")
    frame = pd.DataFrame(records)
    result = pd.DataFrame({
        "funding_time": pd.to_datetime(
            pd.to_numeric(frame.iloc[:, 0], errors="raise") + FOUR_HOURS_MS,
            unit="ms", utc=True,
        ),
        "mark_price": pd.to_numeric(frame.iloc[:, 4], errors="raise"),
    })
    result = result[result["funding_time"].dt.hour.isin([0, 8, 16])]
    return result.drop_duplicates("funding_time", keep="last").sort_values("funding_time")


def load_symbol(symbol, start, end, cache_dir, refresh=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol}_bybit_inputs.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["funding_time"])
    funding = download_funding(symbol, start, end)
    mark = download_mark_klines(symbol, start, end)
    frame = funding.merge(mark, on="funding_time", how="inner")
    frame.to_csv(path, index=False)
    return frame


def build_common_panel(frames):
    pieces = []
    for symbol in sorted(frames):
        frame = frames[symbol].copy().set_index("funding_time").sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate Bybit timestamp for {symbol}")
        if (frame["mark_price"] <= 0).any():
            raise ValueError(f"Nonpositive Bybit mark price for {symbol}")
        pieces.append(frame[["funding_rate", "mark_price"]].rename(columns={
            "funding_rate": f"{symbol}__funding",
            "mark_price": f"{symbol}__mark",
        }))
    panel = pd.concat(pieces, axis=1, join="inner").sort_index()
    if panel.index.has_duplicates:
        raise ValueError("Duplicate common Bybit timestamps")
    return panel


def lagged_rank_weights(panel):
    symbols = sorted(column.split("__")[0] for column in panel if column.endswith("__funding"))
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
    terminal_cost = weights.loc[last_valid].abs().sum() * leg_cost
    trading_cost = turnover * leg_cost
    trading_cost.loc[last_valid] += terminal_cost
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


def diagnostic_labels(detail, oos_start, oos_end):
    oos = detail.loc[oos_start:oos_end]
    daily_spread = oos["lagged_funding_spread"].groupby(oos.index.normalize()).mean()
    median = float(daily_spread.median())
    return pd.DataFrame({
        "funding_spread_regime": np.where(
            daily_spread >= median, "spread_high", "spread_low"
        ),
        "calendar_year": daily_spread.index.year.astype(str),
    }, index=daily_spread.index), median
