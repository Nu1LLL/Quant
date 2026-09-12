"""DIA close-to-open long and open-to-close short OOS implementation."""
from pathlib import Path

import numpy as np
import pandas as pd
import requests


YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/DIA"
LATEST_ACCEPTABLE_START = pd.Timestamp("1998-02-28", tz="UTC")
REQUEST_START = pd.Timestamp("1998-01-01", tz="UTC")
REQUEST_END_EXCLUSIVE = pd.Timestamp("2026-09-14", tz="UTC")


def download_ohlc(session=None):
    client = session or requests.Session()
    response = client.get(
        YAHOO_URL,
        params={
            "period1": int(REQUEST_START.timestamp()),
            "period2": int(REQUEST_END_EXCLUSIVE.timestamp()),
            "interval": "1d", "events": "div,splits",
        },
        headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
    )
    response.raise_for_status()
    result = response.json().get("chart", {}).get("result")
    if not result:
        raise ValueError("Yahoo returned no DIA history")
    result = result[0]
    quote = result["indicators"]["quote"][0]
    adjusted = result.get("indicators", {}).get("adjclose")
    if not adjusted or "adjclose" not in adjusted[0]:
        raise ValueError("Yahoo returned no DIA adjusted close")
    frame = pd.DataFrame({
        "date": pd.to_datetime(result["timestamp"], unit="s", utc=True).normalize(),
        "raw_open": quote["open"],
        "raw_close": quote["close"],
        "adjusted_close": adjusted[0]["adjclose"],
    })
    return frame.dropna().drop_duplicates("date", keep="last").set_index("date")


def load_ohlc(cache_dir, refresh=False, session=None):
    path = Path(cache_dir) / "DIA_1998-01-01_2026-09-13_ohlc_adjusted.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"]).set_index("date")
    frame = download_ohlc(session=session)
    frame.rename_axis("date").to_csv(path)
    return frame


def validate_ohlc(frame):
    frame = pd.DataFrame(frame).copy().sort_index()
    required = ["raw_open", "raw_close", "adjusted_close"]
    if any(column not in frame for column in required):
        raise ValueError(f"DIA data must contain {required}")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame[required].dropna()
    values = frame.to_numpy(dtype=float)
    if frame.empty or frame.index.has_duplicates or not np.isfinite(values).all():
        raise ValueError("DIA OHLC data are empty, duplicated or non-finite")
    if (values <= 0).any():
        raise ValueError("DIA OHLC data must be positive")
    gaps = frame.index.to_series().diff().dropna().dt.days
    if not gaps.empty and int(gaps.max()) > 10:
        raise ValueError("DIA history contains a gap longer than ten days")
    return frame


def return_components(frame):
    frame = validate_ohlc(frame)
    adjustment = frame["adjusted_close"] / frame["raw_close"]
    adjusted_open = frame["raw_open"] * adjustment
    overnight = adjusted_open.div(frame["adjusted_close"].shift(1)).sub(1.0)
    intraday = frame["adjusted_close"].div(adjusted_open).sub(1.0)
    result = pd.DataFrame({
        "adjustment_factor": adjustment,
        "adjusted_open": adjusted_open,
        "overnight_return": overnight,
        "intraday_return": intraday,
    }).dropna()
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("DIA return components contain non-finite values")
    return result


def run_scenario(
    frame, leverage, fill_cost=0.0001, annual_short_borrow=0.005,
    annual_financing=0.04,
):
    components = return_components(frame)
    financing_half = max(leverage - 1.0, 0.0) * annual_financing / 504.0
    borrow_half = leverage * annual_short_borrow / 504.0
    overnight_net = (
        leverage * components["overnight_return"]
        - 2.0 * leverage * fill_cost - financing_half
    )
    intraday_net = (
        -leverage * components["intraday_return"]
        - 2.0 * leverage * fill_cost - financing_half - borrow_half
    )
    net_return = (1.0 + overnight_net) * (1.0 + intraday_net) - 1.0
    detail = components.assign(
        overnight_net=overnight_net,
        intraday_net=intraday_net,
        trading_cost=4.0 * leverage * fill_cost,
        short_borrow=borrow_half,
        financing_cost=2.0 * financing_half,
        net_return=net_return,
    )
    detail["equity"] = 10000.0 * (1.0 + detail["net_return"]).cumprod()
    return detail["net_return"], detail
