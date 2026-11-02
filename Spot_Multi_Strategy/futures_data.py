from pathlib import Path

import pandas as pd
import requests

from data import (
    repair_small_time_gaps,
    to_utc_timestamp,
    validate_ohlcv
)


BINANCE_FUTURES_KLINES_URL = (
    "https://fapi.binance.com/fapi/v1/klines"
)

BINANCE_FUNDING_RATE_URL = (
    "https://fapi.binance.com/fapi/v1/fundingRate"
)


def download_futures_klines(
    symbol,
    interval,
    start_time,
    end_time,
    timeout=15,
    session=None
):
    # 下载Binance U本位永续合约的公开K线，不需要API密钥
    start_datetime = to_utc_timestamp(start_time)
    end_datetime = to_utc_timestamp(end_time)

    if start_datetime >= end_datetime:
        raise ValueError("开始时间必须早于结束时间")

    request_session = session or requests.Session()
    start_time_ms = int(start_datetime.timestamp() * 1000)
    end_time_ms = int(end_datetime.timestamp() * 1000)
    all_data = []

    while start_time_ms < end_time_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_time_ms,
            "endTime": end_time_ms - 1,
            "limit": 1500
        }

        response = request_session.get(
            BINANCE_FUTURES_KLINES_URL,
            params=params,
            timeout=timeout
        )
        response.raise_for_status()
        batch_data = response.json()

        if not batch_data:
            break

        all_data.extend(batch_data)
        next_start_time = int(batch_data[-1][0]) + 1

        if next_start_time <= start_time_ms:
            raise RuntimeError("合约K线分页时间没有向前移动")

        start_time_ms = next_start_time

        if len(batch_data) < 1500:
            break

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]

    df = pd.DataFrame(all_data, columns=columns)

    if df.empty:
        raise ValueError("该交易品种和时间范围没有返回合约K线")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    df[numeric_columns] = df[numeric_columns].astype(float)
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )
    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True
    )

    # 只保留固定区间内已经结束的K线
    df = df[
        (df["open_time"] >= start_datetime)
        &
        (df["close_time"] < end_datetime)
    ].copy()

    df = df.drop_duplicates(subset=["open_time"])

    return repair_small_time_gaps(
        original_df=df,
        interval=interval
    )


def load_or_download_futures_klines(
    symbol,
    interval,
    start_time,
    end_time,
    cache_folder="futures_data_cache",
    refresh=False
):
    # 优先读取本地缓存，避免重复请求公开行情接口
    start_datetime = to_utc_timestamp(start_time)
    end_datetime = to_utc_timestamp(end_time)
    cache_path = Path(cache_folder)
    cache_path.mkdir(parents=True, exist_ok=True)

    safe_start = start_datetime.strftime("%Y%m%d")
    safe_end = end_datetime.strftime("%Y%m%d")
    data_file = cache_path / (
        f"{symbol.upper()}_{interval}_{safe_start}_{safe_end}_futures.csv"
    )

    if data_file.exists() and not refresh:
        df = pd.read_csv(data_file, parse_dates=["open_time"])

        return validate_ohlcv(
            df,
            interval=interval,
            strict_continuity=True
        )

    df = download_futures_klines(
        symbol=symbol,
        interval=interval,
        start_time=start_datetime,
        end_time=end_datetime
    )
    df.to_csv(data_file, index=False)
    return df


def validate_funding_rates(original_df):
    # 检查资金费率字段并按照时间排序
    required_columns = [
        "symbol",
        "funding_time",
        "funding_rate",
        "mark_price"
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in original_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"资金费率缺少必要字段：{missing_columns}"
        )

    df = original_df[required_columns].copy()

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["funding_time"] = pd.to_datetime(
        df["funding_time"],
        utc=True,
        format="mixed"
    )
    df["funding_rate"] = pd.to_numeric(
        df["funding_rate"],
        errors="raise"
    )
    df["mark_price"] = pd.to_numeric(
        df["mark_price"],
        errors="raise"
    )

    if df[required_columns].isna().any().any():
        raise ValueError("资金费率中存在空值")

    return (
        df.drop_duplicates(subset=["symbol", "funding_time"])
        .sort_values("funding_time")
        .reset_index(drop=True)
    )


def download_funding_rates(
    symbol,
    start_time,
    end_time,
    timeout=15,
    session=None
):
    # 下载Binance U本位永续合约的公开历史资金费率
    start_datetime = to_utc_timestamp(start_time)
    end_datetime = to_utc_timestamp(end_time)

    if start_datetime >= end_datetime:
        raise ValueError("开始时间必须早于结束时间")

    request_session = session or requests.Session()
    start_time_ms = int(start_datetime.timestamp() * 1000)
    end_time_ms = int(end_datetime.timestamp() * 1000)
    all_data = []

    while start_time_ms < end_time_ms:
        params = {
            "symbol": symbol.upper(),
            "startTime": start_time_ms,
            "endTime": end_time_ms - 1,
            "limit": 1000
        }

        response = request_session.get(
            BINANCE_FUNDING_RATE_URL,
            params=params,
            timeout=timeout
        )
        response.raise_for_status()
        batch_data = response.json()

        if not batch_data:
            break

        all_data.extend(batch_data)
        next_start_time = int(batch_data[-1]["fundingTime"]) + 1

        if next_start_time <= start_time_ms:
            raise RuntimeError("资金费率分页时间没有向前移动")

        start_time_ms = next_start_time

        if len(batch_data) < 1000:
            break

    rows = [
        {
            "symbol": item["symbol"],
            "funding_time": pd.to_datetime(
                item["fundingTime"],
                unit="ms",
                utc=True
            ),
            "funding_rate": float(item["fundingRate"]),
            "mark_price": float(item.get("markPrice", 0) or 0)
        }
        for item in all_data
    ]

    return validate_funding_rates(
        pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "funding_time",
                "funding_rate",
                "mark_price"
            ]
        )
    )


def load_or_download_funding_rates(
    symbol,
    start_time,
    end_time,
    cache_folder="futures_data_cache",
    refresh=False
):
    # 读取或下载与回测区间对应的历史资金费率
    start_datetime = to_utc_timestamp(start_time)
    end_datetime = to_utc_timestamp(end_time)
    cache_path = Path(cache_folder)
    cache_path.mkdir(parents=True, exist_ok=True)

    safe_start = start_datetime.strftime("%Y%m%d")
    safe_end = end_datetime.strftime("%Y%m%d")
    data_file = cache_path / (
        f"{symbol.upper()}_{safe_start}_{safe_end}_funding.csv"
    )

    if data_file.exists() and not refresh:
        df = pd.read_csv(
            data_file,
            parse_dates=["funding_time"]
        )
        return validate_funding_rates(df)

    df = download_funding_rates(
        symbol=symbol,
        start_time=start_datetime,
        end_time=end_datetime
    )
    df.to_csv(data_file, index=False)
    return df
