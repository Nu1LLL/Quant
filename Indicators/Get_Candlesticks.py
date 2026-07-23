import requests
import pandas as pd


def get_candlesticks(
    symbol="BTCUSDT",
    interval="15m",
    days=1,
    end_time=None
):  # 获取指定交易品种和时间范围的K线

    url = "https://data-api.binance.vision/api/v3/klines"

    if end_time is None:

        end_datetime = pd.Timestamp.now(
            tz="UTC"
        )
        # 没有指定结束时间时，使用当前UTC时间

    else:

        end_datetime = pd.Timestamp(
            end_time
        )
        # 把传入的结束时间转换成Pandas时间


        if end_datetime.tzinfo is None: 

            end_datetime = end_datetime.tz_localize(
                "UTC"
            )
            # 没有时区时，默认把它当成UTC时间


        else:

            end_datetime = end_datetime.tz_convert(
                "UTC"
            )
            # 有时区时，统一转换成UTC时间


    start_datetime = (
        end_datetime
        -
        pd.Timedelta(days=days)
    )
    # 根据结束时间和天数计算开始时间


    end_time_ms = int(end_datetime.timestamp() * 1000)
    start_time_ms = int(start_datetime.timestamp() * 1000)

    all_data = []

    while start_time_ms < end_time_ms:

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "limit": 1000
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        batch_data = response.json()

        if len(batch_data) == 0:
            break

        all_data.extend(batch_data)

    
        # f"本次获取：{len(batch_data)}根，"
        # f"累计：{len(all_data)}根",
        # flush=True
        #测试程序是否正确

        next_start_time = batch_data[-1][0] + 1

        # 防止异常数据导致无限循环
        if next_start_time <= start_time_ms:
            raise RuntimeError("K线时间没有向前移动")

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

    # 删除尚未结束的当前K线
    df = df[
        df["close_time"] <= end_datetime
    ].copy()

    # 防止出现重复K线
    df = df.drop_duplicates(
        subset=["open_time"]
    )

    # 确保按照时间从旧到新排列
    df = df.sort_values("open_time")

    # 重新生成连续索引
    df = df.reset_index(drop=True)

    return df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]


if __name__ == "__main__":
#   测试获取K线数据

    test_df = get_candlesticks(
        symbol="BTCUSDT",
        interval="4h",
        days=10
    )

    print("\n获取完成")
    print(f"K线数量：{len(test_df)}")
    print(test_df.head())
    print(test_df.tail())