"""Binance perpetual-futures funding data adapter.

This module restores the optional data adapter expected by alpha_research.py.
Funding observations are indexed by their settlement time; callers must not
forward-fill them to times before that settlement is known.
"""

from pathlib import Path

import pandas as pd
import requests

from data import to_utc_timestamp


BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def validate_funding_rates(original_df):
    df = original_df.copy()
    required = ["funding_time", "funding_rate"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"funding data missing columns: {missing}")
    if df.empty:
        raise ValueError("no funding-rate observations")
    df["funding_time"] = pd.to_datetime(
        df["funding_time"], utc=True, format="mixed"
    )
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="raise")
    if df[required].isna().any().any():
        raise ValueError("funding data contains null values")
    if df["funding_time"].duplicated().any():
        raise ValueError("funding data contains duplicate timestamps")
    return df.sort_values("funding_time").reset_index(drop=True)[required]


def download_funding_rates(
    symbol, start_time, end_time, timeout=15, session=None
):
    start = to_utc_timestamp(start_time)
    end = to_utc_timestamp(end_time)
    if start >= end:
        raise ValueError("start_time must precede end_time")
    client = session or requests.Session()
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    records = []
    while cursor_ms < end_ms:
        response = client.get(
            BINANCE_FUNDING_URL,
            params={
                "symbol": symbol.upper(), "startTime": cursor_ms,
                "endTime": end_ms - 1, "limit": 1000,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        records.extend(batch)
        next_cursor = int(batch[-1]["fundingTime"]) + 1
        if next_cursor <= cursor_ms:
            raise RuntimeError("funding-rate pagination did not advance")
        cursor_ms = next_cursor
        if len(batch) < 1000:
            break
    if not records:
        raise ValueError("no funding rates returned for requested range")
    frame = pd.DataFrame(
        {
            "funding_time": pd.to_datetime(
                [row["fundingTime"] for row in records], unit="ms", utc=True
            ),
            "funding_rate": [float(row["fundingRate"]) for row in records],
        }
    )
    frame = frame[(frame["funding_time"] >= start) & (frame["funding_time"] < end)]
    return validate_funding_rates(frame)


def load_or_download_funding_rates(
    symbol, start_time, end_time, cache_folder="futures_data_cache",
    force_download=False, session=None
):
    start = to_utc_timestamp(start_time)
    end = to_utc_timestamp(end_time)
    folder = Path(cache_folder)
    folder.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{symbol.upper()}_funding_{start.strftime('%Y%m%d')}_"
        f"{end.strftime('%Y%m%d')}.csv"
    )
    cache_path = folder / filename
    if cache_path.exists() and not force_download:
        return validate_funding_rates(pd.read_csv(cache_path))
    frame = download_funding_rates(
        symbol, start, end, session=session
    )
    frame.to_csv(cache_path, index=False)
    return frame


def attach_funding_rates_to_bars(bars, funding_rates):
    """Attach each settlement to the bar interval containing its timestamp.

    Binance occasionally returns settlement timestamps a few milliseconds after
    the nominal hour.  An exact timestamp join silently drops those real cash
    flows from minute bars, so we align each observation down to the inferred
    bar boundary without carrying it into any later bar.
    """
    output = bars.copy()
    output["open_time"] = pd.to_datetime(output["open_time"], utc=True)
    if len(output) < 2:
        raise ValueError("at least two bars are required to align funding")
    bar_interval = output["open_time"].sort_values().diff().dropna().median()
    if pd.isna(bar_interval) or bar_interval <= pd.Timedelta(0):
        raise ValueError("unable to infer a positive bar interval")
    funding = validate_funding_rates(funding_rates)
    funding["open_time"] = funding["funding_time"].dt.floor(bar_interval)
    funding = (
        funding.groupby("open_time", as_index=False)["funding_rate"]
        .sum()
    )
    output = output.merge(funding, on="open_time", how="left")
    output["funding_rate"] = output["funding_rate"].fillna(0.0)
    return output
