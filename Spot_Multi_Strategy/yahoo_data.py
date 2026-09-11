"""美股日线数据获取（Yahoo Finance公开chart接口，不需要API密钥）。

只是数据源不同，产出的列名（open_time/open/high/low/close/volume）
和data.py（Binance现货）完全一致，可以直接喂给alphas/、walk_forward.py、
market_neutral.py等所有现有管线，不需要改动那些代码。

Yahoo这个接口是非官方的公开endpoint（雅虎自己网页用的同一个接口），
不是付费API，本模块只做只读的历史行情查询，不涉及账户、下单。
"""
from pathlib import Path

import pandas as pd
import requests

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0"


def download_yahoo_daily_klines(symbol, range_="10y", timeout=15, session=None):
    request_session = session or requests.Session()
    response = request_session.get(
        YAHOO_CHART_URL.format(symbol=symbol),
        params={"range": range_, "interval": "1d"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()

    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"Yahoo Finance没有返回{symbol}的数据")

    result = result[0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "open_time": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote["volume"]
    })

    # Yahoo偶尔会在停牌日返回全NaN的行，直接丢弃，不做任何填补/编造
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(
        drop=True
    )
    return df


def load_or_download_yahoo_klines(
    symbol,
    range_="10y",
    cache_folder="yahoo_data_cache",
    refresh=False
):
    cache_path = Path(cache_folder)
    cache_path.mkdir(parents=True, exist_ok=True)
    data_file = cache_path / f"{symbol.upper()}_{range_}_1d.csv"

    if data_file.exists() and not refresh:
        return pd.read_csv(data_file, parse_dates=["open_time"])

    df = download_yahoo_daily_klines(symbol, range_=range_)
    df.to_csv(data_file, index=False)
    return df
