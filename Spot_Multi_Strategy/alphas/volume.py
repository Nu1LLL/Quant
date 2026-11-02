"""A10 成交量异常、A11 签名成交量压力代理、A12 收盘位置值、
A13 K线内位置、A14 跳空代理。所有信号只使用OHLCV蜡烛数据，
没有真实逐笔委托流，凡是订单流相关的都在metadata里明确标注为代理指标。
"""
import numpy as np

from . import base


def build_volume_surprise(df, window=48):
    volume = df["volume"]
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std(ddof=0)
    volume_zscore = (volume - mean) / std.replace(0, np.nan)

    price_direction = np.sign(df["close"] - df["open"])
    combined = volume_zscore.clip(-3, 3) * price_direction
    normalized = base.squash(combined, scale=3.0)

    name = f"A10_volume_surprise_{window}"
    return {
        name: base.make_signal(
            name=name,
            raw=combined,
            normalized=normalized,
            direction="trend",
            lookback=window + 1,
            window=window,
            rationale=(
                "成交量z-score结合当根K线涨跌方向，检验异常放量"
                "是否伴随价格方向能预测后续走势，而不是把成交量本身"
                "当作独立alpha。"
            )
        )
    }


def build_signed_volume_pressure(df, smooth_window=6):
    close, high, low, volume = (
        df["close"], df["high"], df["low"], df["volume"]
    )
    range_ = (high - low).replace(0, np.nan)
    close_location_value = ((close - low) - (high - close)) / range_
    signed_volume = volume * close_location_value.fillna(0.0)

    smoothed = signed_volume.rolling(smooth_window).mean()
    scale = volume.rolling(
        smooth_window * 4,
        min_periods=smooth_window
    ).mean().replace(0, np.nan)
    raw = smoothed / scale
    normalized = base.squash(raw, scale=1.0)

    name = f"A11_signed_volume_pressure_{smooth_window}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="trend",
            lookback=smooth_window * 4 + 1,
            smooth_window=smooth_window,
            is_order_flow_proxy=True,
            rationale=(
                "用收盘位置加权成交量构造的OHLCV订单流代理指标"
                "（明确不是真实逐笔委托流），平滑后衡量买卖压力。"
            )
        )
    }


def build_close_location_value(df, window=10):
    high, low, close = df["high"], df["low"], df["close"]
    range_ = (high - low).replace(0, np.nan)
    clv = (((close - low) - (high - close)) / range_).fillna(0.0)
    raw = clv.rolling(window).mean()

    name = f"A12_close_location_value_{window}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=raw.clip(-1, 1),
            direction="trend",
            lookback=window + 1,
            window=window,
            rationale=(
                "持续收盘于K线高点附近可能预示买方主导和动量延续，"
                "滚动平均收盘位置值(CLV)。"
            )
        )
    }


def build_intraday_range_position(df, window=10):
    high, low, close = df["high"], df["low"], df["close"]
    range_ = (high - low).replace(0, np.nan)
    position = ((close - low) / range_).fillna(0.5)
    raw = position.rolling(window).mean()
    normalized = (raw - 0.5) * 2.0

    name = f"A13_intraday_range_position_{window}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="trend",
            lookback=window + 1,
            window=window,
            rationale=(
                "收盘价在当根K线高低点区间中的相对位置，滚动平均后"
                "衡量买盘/卖盘主导程度，避免除以零用0.5（区间中点）填充。"
            )
        )
    }


def build_gap_proxy(df):
    open_price, prior_close = df["open"], df["close"].shift(1)
    raw = open_price / prior_close - 1
    normalized = base.squash(raw, scale=0.01)

    name = "A14_gap_proxy"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="trend",
            lookback=2,
            needs_significance_check=True,
            rationale=(
                "4小时线没有真正的隔夜跳空，但开盘价相对上一根收盘价"
                "的跳变仍可能包含信息（例如场外/其他交易所引导）。"
                "只有在alpha_research.py中验证扣费后有统计显著性时"
                "才应保留，否则应在研究报告中标注拒绝原因。"
            )
        )
    }


def build_alphas(df):
    alphas = {}
    alphas.update(build_volume_surprise(df))
    alphas.update(build_signed_volume_pressure(df))
    alphas.update(build_close_location_value(df))
    alphas.update(build_intraday_range_position(df))
    alphas.update(build_gap_proxy(df))
    return alphas
