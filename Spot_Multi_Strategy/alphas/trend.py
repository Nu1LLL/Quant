"""A02 EMA距离动量、A03 Donchian突破强度、A08 趋势质量、A09 效率比率。"""
import numpy as np
import pandas as pd

from indicators import calculate_adx, calculate_atr

from . import base

DEFAULT_EMA_WINDOWS = (20, 50, 100)


def build_ema_distance_momentum(
    df,
    ema_windows=DEFAULT_EMA_WINDOWS,
    atr_window=14
):
    close = df["close"]
    atr = calculate_atr(df, window=atr_window)

    signals = {}
    for window in ema_windows:
        ema = close.ewm(
            span=window,
            adjust=False,
            min_periods=window
        ).mean()
        raw = (close - ema) / atr.replace(0, np.nan)
        normalized = base.squash(raw, scale=3.0)

        name = f"A02_ema_distance_{window}"
        signals[name] = base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="trend",
            lookback=max(window, atr_window) + 1,
            ema_window=window,
            rationale=(
                f"价格相对{window}周期EMA的ATR标准化距离，连续衡量"
                "趋势强度，而不是只用收盘价>EMA的布尔条件。"
            )
        )

    return signals


def build_donchian_breakout_strength(df, entry_window=20, atr_window=14):
    high, low, close = df["high"], df["low"], df["close"]
    atr = calculate_atr(df, window=atr_window)

    prior_high = high.rolling(entry_window).max().shift(1)
    prior_low = low.rolling(entry_window).min().shift(1)

    upside = (close - prior_high) / atr.replace(0, np.nan)
    downside = (close - prior_low) / atr.replace(0, np.nan)

    raw = pd.Series(
        np.where(
            upside > 0,
            upside,
            np.where(downside < 0, downside, 0.0)
        ),
        index=df.index
    )
    normalized = base.squash(raw, scale=2.0)

    name = f"A03_donchian_breakout_{entry_window}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="trend",
            lookback=max(entry_window, atr_window) + 1,
            entry_window=entry_window,
            rationale=(
                "以ATR标准化衡量收盘价突破前N根K线高/低点的强度。"
                "现货只做多，向下突破在组合层面代表降低仓位，而不是做空。"
            )
        )
    }


def _efficiency_ratio(close, window):
    net_change = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return (net_change / path_length.replace(0, np.nan)).clip(0, 1)


def build_trend_quality(
    df,
    ema_window=50,
    ema_slope_window=10,
    adx_window=14,
    atr_window=14,
    efficiency_window=20
):
    close = df["close"]
    atr = calculate_atr(df, window=atr_window)
    ema = close.ewm(
        span=ema_window,
        adjust=False,
        min_periods=ema_window
    ).mean()
    ema_slope = (
        (ema - ema.shift(ema_slope_window)) / atr.replace(0, np.nan)
    )

    adx_frame = calculate_adx(df, window=adx_window)
    directional_diff = (
        (adx_frame["plus_di"] - adx_frame["minus_di"]) / 100.0
    ).clip(-1, 1)
    adx_component = (adx_frame["adx"] / 50.0).clip(upper=1.5) - 0.5

    efficiency = _efficiency_ratio(close, efficiency_window) - 0.5

    raw = (
        base.squash(ema_slope, scale=2.0)
        + directional_diff
        + adx_component
        + efficiency
    ) / 4.0

    lookback = max(
        ema_window + ema_slope_window,
        adx_window,
        efficiency_window
    ) + 1

    name = "A08_trend_quality"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=raw,
            direction="trend",
            lookback=lookback,
            rationale=(
                "组合EMA斜率(ATR标准化)、ADX水平、+DI-DI方向差与"
                "Kaufman效率比率，衡量趋势质量，与既有的单一ADX阈值"
                "regime过滤互补而非重复。"
            )
        )
    }


def build_price_efficiency_ratio(df, window=20):
    close = df["close"]
    efficiency_ratio = _efficiency_ratio(close, window)
    direction_sign = np.sign(close - close.shift(window))
    raw = efficiency_ratio * direction_sign

    name = f"A09_efficiency_ratio_{window}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=raw,
            direction="trend",
            lookback=window + 1,
            window=window,
            regime_feature=True,
            rationale=(
                "Kaufman效率比率衡量方向性/噪音比：高值代表趋势主导、"
                "低值代表震荡。既作为独立alpha（乘以近期涨跌方向），"
                "也作为regime特征供后续研究使用。"
            )
        )
    }


def build_alphas(df):
    alphas = {}
    alphas.update(build_ema_distance_momentum(df))
    alphas.update(build_donchian_breakout_strength(df))
    alphas.update(build_trend_quality(df))
    alphas.update(build_price_efficiency_ratio(df))
    return alphas
