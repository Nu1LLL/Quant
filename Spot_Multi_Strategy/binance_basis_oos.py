"""Public Binance spot/perpetual data and delta-neutral basis-carry model."""
from pathlib import Path

import numpy as np
import pandas as pd
import requests


SPOT_URL = "https://data-api.binance.vision/api/v3/klines"
MARK_URL = "https://fapi.binance.com/fapi/v1/markPriceKlines"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
EIGHT_HOURS_MS = 8 * 60 * 60 * 1000


def _milliseconds(value):
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)


def download_klines(symbol, start, end, mark=False, session=None):
    client = session or requests.Session()
    url = MARK_URL if mark else SPOT_URL
    cursor, end_ms = _milliseconds(start), _milliseconds(end)
    records = []
    while cursor <= end_ms:
        response = client.get(url, params={
            "symbol": symbol, "interval": "8h", "startTime": cursor,
            "endTime": end_ms, "limit": 1000,
        }, timeout=30)
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        records.extend(page)
        next_cursor = int(page[-1][0]) + EIGHT_HOURS_MS
        if next_cursor <= cursor:
            raise RuntimeError("Binance kline pagination did not advance")
        cursor = next_cursor
        if len(page) < 1000:
            break
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No {'mark' if mark else 'spot'} klines for {symbol}")
    result = pd.DataFrame({
        "funding_time": pd.to_datetime(frame.iloc[:, 0], unit="ms", utc=True),
        "price": pd.to_numeric(frame.iloc[:, 1], errors="raise"),
    }).drop_duplicates("funding_time", keep="last").sort_values("funding_time")
    return result


def download_funding(symbol, start, end, session=None):
    client = session or requests.Session()
    cursor, end_ms = _milliseconds(start), _milliseconds(end)
    records = []
    while cursor <= end_ms:
        response = client.get(FUNDING_URL, params={
            "symbol": symbol, "startTime": cursor, "endTime": end_ms,
            "limit": 1000,
        }, timeout=30)
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        records.extend(page)
        next_cursor = int(page[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("Binance funding pagination did not advance")
        cursor = next_cursor
        if len(page) < 1000:
            break
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No funding history for {symbol}")
    raw_time = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
    scheduled_time = raw_time.dt.round("8h")
    close_to_schedule = (raw_time - scheduled_time).abs() <= pd.Timedelta(minutes=5)
    result = pd.DataFrame({
        "funding_time": scheduled_time[close_to_schedule],
        "funding_rate": pd.to_numeric(frame["fundingRate"], errors="raise"),
    }).dropna().drop_duplicates("funding_time", keep="last").sort_values("funding_time")
    return result


def load_symbol(symbol, start, end, cache_dir, refresh=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol}_basis_inputs.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["funding_time"])
    spot = download_klines(symbol, start, end, mark=False).rename(columns={"price": "spot"})
    mark = download_klines(symbol, start, end, mark=True).rename(columns={"price": "perp_mark"})
    funding = download_funding(symbol, start, end)
    frame = spot.merge(mark, on="funding_time", how="inner").merge(
        funding, on="funding_time", how="inner"
    )
    frame = frame[frame["funding_time"].dt.hour.isin([0, 8, 16])]
    frame.to_csv(path, index=False)
    return frame


def build_interval_components(frames):
    components = []
    for symbol, source in frames.items():
        frame = source.copy().set_index("funding_time").sort_index()
        spot_return = frame["spot"].pct_change(fill_method=None)
        perp_return = frame["perp_mark"].pct_change(fill_method=None)
        components.append(pd.DataFrame({
            f"{symbol}__gross": spot_return - perp_return + frame["funding_rate"],
            f"{symbol}__hedge_turnover": spot_return.abs() + perp_return.abs(),
            f"{symbol}__funding": frame["funding_rate"],
            f"{symbol}__spot": frame["spot"],
        }))
    result = pd.concat(components, axis=1, join="inner").dropna()
    if result.index.has_duplicates:
        raise ValueError("Duplicate common funding timestamps")
    return result


def run_notional(intervals, leverage, leg_cost=0.0005, annual_financing=0.04):
    symbols = sorted(column.split("__")[0] for column in intervals if column.endswith("__gross"))
    gross = pd.concat(
        [intervals[f"{symbol}__gross"] for symbol in symbols], axis=1
    ).mean(axis=1)
    hedge_turnover = pd.concat(
        [intervals[f"{symbol}__hedge_turnover"] for symbol in symbols], axis=1
    ).mean(axis=1)
    trading_cost = leverage * hedge_turnover * leg_cost
    trading_cost.iloc[0] += leverage * 2.0 * leg_cost
    financing = max(leverage - 1.0, 0.0) * annual_financing / 1095.0
    interval_return = leverage * gross - trading_cost - financing
    daily_return = interval_return.groupby(interval_return.index.normalize()).apply(
        lambda values: float((1.0 + values).prod() - 1.0)
    )
    daily_return.name = "net_return"
    detail = pd.DataFrame({
        "gross_pair_return": gross,
        "hedge_turnover": hedge_turnover,
        "trading_cost": trading_cost,
        "financing_cost": financing,
        "net_return": interval_return,
    })
    return daily_return, detail


def diagnostic_regimes(intervals):
    funding_columns = [c for c in intervals if c.endswith("__funding")]
    average_funding = intervals[funding_columns].mean(axis=1)
    daily_funding = average_funding.groupby(average_funding.index.normalize()).mean()
    btc = intervals["BTCUSDT__spot"].groupby(intervals.index.normalize()).last()
    btc_trend = (btc > btc.rolling(200, min_periods=200).mean()).shift(1)
    return pd.DataFrame({
        "funding_regime": np.where(daily_funding >= 0, "funding_positive", "funding_negative"),
        "btc_trend_regime": btc_trend.map({True: "btc_above_200d", False: "btc_below_200d"}),
    }, index=daily_funding.index)
