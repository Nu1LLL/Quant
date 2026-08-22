from pathlib import Path

import pandas as pd
import requests


BINANCE_KLINES_URL = (
    "https://data-api.binance.vision/api/v3/klines"
)


INTERVAL_TO_TIMEDELTA = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1)
}


def to_utc_timestamp(value):
    # 把字符串或时间对象统一转换为UTC时间
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def validate_ohlcv(
    original_df,
    interval,
    strict_continuity=True
):
    # 复制数据，防止检查过程修改调用者的数据
    df = original_df.copy()

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"K线缺少必要字段：{missing_columns}"
        )

    if df.empty:
        raise ValueError("没有获取到任何K线")

    if interval not in INTERVAL_TO_TIMEDELTA:
        raise ValueError(f"暂不支持K线周期：{interval}")

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="raise"
    )

    if df[required_columns].isna().any().any():
        raise ValueError("K线中存在空值")

    if df["open_time"].duplicated().any():
        raise ValueError("K线中存在重复时间")

    df = df.sort_values("open_time").reset_index(drop=True)

    invalid_price = (
        (df[["open", "high", "low", "close"]] <= 0)
        .any(axis=1)
    )

    if invalid_price.any():
        raise ValueError("K线中存在小于或等于0的价格")

    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("K线最高价字段不合法")

    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("K线最低价字段不合法")

    if (df["volume"] < 0).any():
        raise ValueError("K线成交量不能小于0")

    if strict_continuity and len(df) > 1:
        expected_delta = INTERVAL_TO_TIMEDELTA[interval]
        actual_delta = df["open_time"].diff().dropna()
        gap_mask = actual_delta != expected_delta

        if gap_mask.any():
            first_gap_position = gap_mask[gap_mask].index[0]
            raise ValueError(
                "K线时间不连续，首次缺口位于："
                f"{df.loc[first_gap_position, 'open_time']}"
            )

    output_columns = required_columns.copy()

    if "is_synthetic" in df.columns:
        df["is_synthetic"] = df["is_synthetic"].astype(bool)
        output_columns.append("is_synthetic")

    return df[output_columns]


def repair_small_time_gaps(
    original_df,
    interval,
    maximum_consecutive_missing=6
):
    # 对交易所短暂停机造成的小缺口补零成交量K线
    df = validate_ohlcv(
        original_df,
        interval=interval,
        strict_continuity=False
    )

    expected_delta = INTERVAL_TO_TIMEDELTA[interval]
    full_time_index = pd.date_range(
        start=df["open_time"].iloc[0],
        end=df["open_time"].iloc[-1],
        freq=expected_delta,
        tz="UTC"
    )

    indexed_df = df.set_index("open_time").reindex(full_time_index)
    missing_mask = indexed_df["close"].isna()

    if not missing_mask.any():
        indexed_df["is_synthetic"] = False
    else:
        missing_groups = (
            missing_mask.ne(missing_mask.shift()).cumsum()
        )
        largest_gap = int(
            missing_mask.groupby(missing_groups).sum().max()
        )

        if largest_gap > maximum_consecutive_missing:
            raise ValueError(
                "K线存在过大的连续缺口，最多允许自动修复"
                f"{maximum_consecutive_missing}根，实际为{largest_gap}根"
            )

        previous_close = indexed_df["close"].ffill()

        for column in ["open", "high", "low", "close"]:
            indexed_df.loc[missing_mask, column] = (
                previous_close.loc[missing_mask]
            )

        indexed_df.loc[missing_mask, "volume"] = 0.0
        indexed_df["is_synthetic"] = missing_mask

    indexed_df.index.name = "open_time"
    repaired_df = indexed_df.reset_index()

    return validate_ohlcv(
        repaired_df,
        interval=interval,
        strict_continuity=True
    )


def download_klines(
    symbol,
    interval,
    start_time,
    end_time,
    timeout=15,
    session=None
):
    # 下载指定固定时间范围的Binance现货K线
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
            "limit": 1000
        }

        response = request_session.get(
            BINANCE_KLINES_URL,
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
            raise RuntimeError("K线分页时间没有向前移动")

        start_time_ms = next_start_time

        if len(batch_data) < 1000:
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
        raise ValueError(
            "该交易品种和时间范围没有返回K线"
        )

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

    # 只保留已经结束并且位于固定范围内的K线
    df = df[
        (df["open_time"] >= start_datetime)
        &
        (df["close_time"] < end_datetime)
    ].copy()

    df = df.drop_duplicates(
        subset=["open_time"]
    )

    return repair_small_time_gaps(
        original_df=df,
        interval=interval
    )


def load_or_download_klines(
    symbol,
    interval,
    start_time,
    end_time,
    cache_folder="data_cache",
    refresh=False
):
    # 生成包含固定时间范围的缓存文件名
    start_datetime = to_utc_timestamp(start_time)
    end_datetime = to_utc_timestamp(end_time)

    cache_path = Path(cache_folder)
    cache_path.mkdir(parents=True, exist_ok=True)

    safe_start = start_datetime.strftime("%Y%m%d")
    safe_end = end_datetime.strftime("%Y%m%d")

    data_file = cache_path / (
        f"{symbol.upper()}_{interval}_{safe_start}_{safe_end}.csv"
    )

    if data_file.exists() and not refresh:
        df = pd.read_csv(
            data_file,
            parse_dates=["open_time"]
        )

        return validate_ohlcv(
            df,
            interval=interval,
            strict_continuity=True
        )

    df = download_klines(
        symbol=symbol,
        interval=interval,
        start_time=start_datetime,
        end_time=end_datetime
    )

    df.to_csv(data_file, index=False)
    return df
