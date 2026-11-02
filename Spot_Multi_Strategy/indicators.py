import numpy as np
import pandas as pd


def calculate_atr(df, window=14):
    # 获取上一根K线的收盘价
    previous_close = df["close"].shift(1)

    high_low = df["high"] - df["low"]
    high_previous_close = (
        df["high"] - previous_close
    ).abs()
    low_previous_close = (
        df["low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_previous_close,
            low_previous_close
        ],
        axis=1
    ).max(axis=1)

    # 使用Wilder平滑方法计算ATR
    return true_range.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window
    ).mean()


def calculate_rsi(close, window=14):
    # 计算每根K线的收盘价变化
    price_change = close.diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    # 只有上涨没有下跌时RSI为100
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100.0
    )

    # 完全没有价格变化时RSI为50
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50.0
    )

    return rsi


def calculate_adx(df, window=14):
    # 计算上涨方向和下跌方向的单根K线移动距离
    upward_move = df["high"].diff()
    downward_move = -df["low"].diff()

    plus_directional_move = pd.Series(
        np.where(
            (upward_move > downward_move) & (upward_move > 0),
            upward_move,
            0.0
        ),
        index=df.index
    )
    minus_directional_move = pd.Series(
        np.where(
            (downward_move > upward_move) & (downward_move > 0),
            downward_move,
            0.0
        ),
        index=df.index
    )

    atr = calculate_atr(df, window=window)
    smoothed_plus_move = plus_directional_move.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window
    ).mean()
    smoothed_minus_move = minus_directional_move.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window
    ).mean()

    plus_di = 100 * smoothed_plus_move / atr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_move / atr.replace(0, np.nan)
    directional_sum = plus_di + minus_di
    dx = (
        100
        *
        (plus_di - minus_di).abs()
        /
        directional_sum.replace(0, np.nan)
    )
    adx = dx.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window
    ).mean()

    return pd.DataFrame(
        {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di
        },
        index=df.index
    )
