"""A07 波动率压缩/扩张 与 A15 波动率调整突破。"""
from indicators import calculate_atr

from . import base
from .trend import build_donchian_breakout_strength


def build_volatility_regime(
    df,
    short_window=20,
    long_window=100,
    atr_window=14,
    atr_percentile_window=250
):
    close = df["close"]
    short_vol = base.realized_volatility(close, short_window)
    long_vol = base.realized_volatility(close, long_window)
    ratio = short_vol / long_vol.replace(0, float("nan"))

    atr = calculate_atr(df, window=atr_window)
    atr_percentile = atr.rolling(
        atr_percentile_window,
        min_periods=atr_percentile_window
    ).rank(pct=True)

    compression_raw = (1.0 - ratio).clip(-1, 1)
    compression_name = "A07_volatility_compression"

    percentile_name = "A07_atr_percentile"
    percentile_normalized = (atr_percentile - 0.5) * 2.0

    return {
        compression_name: base.make_signal(
            name=compression_name,
            raw=compression_raw,
            normalized=compression_raw,
            direction="regime",
            lookback=max(long_window, short_window) + 1,
            rationale=(
                "短期已实现波动率相对长期波动率的压缩程度"
                "（1-短期/长期），研究低波动压缩后突破的可预测性；"
                "本身不代表方向，是regime特征。"
            )
        ),
        percentile_name: base.make_signal(
            name=percentile_name,
            raw=atr_percentile,
            normalized=percentile_normalized,
            direction="regime",
            lookback=atr_percentile_window + atr_window + 1,
            rationale=(
                "ATR在最近窗口内的分位数，衡量当前波动率所处的"
                "历史位置，作为regime特征使用。"
            )
        )
    }


def build_volatility_adjusted_breakout(
    df,
    entry_window=20,
    atr_window=14,
    vol_window=100
):
    breakout_signals = build_donchian_breakout_strength(
        df,
        entry_window=entry_window,
        atr_window=atr_window
    )
    breakout = breakout_signals[f"A03_donchian_breakout_{entry_window}"]

    close = df["close"]
    vol = base.realized_volatility(close, vol_window)
    vol_rank = vol.rolling(
        vol_window,
        min_periods=vol_window
    ).rank(pct=True)

    # 波动率越低，同样的突破距离获得越高权重（结构性突破更可信）
    vol_weight = (1.0 - vol_rank).clip(0.2, 1.0)
    raw = breakout.raw_signal * vol_weight
    normalized = base.squash(raw, scale=2.0)

    name = f"A15_volatility_adjusted_breakout_{entry_window}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="trend",
            lookback=max(breakout.lookback, vol_window) + 1,
            entry_window=entry_window,
            rationale=(
                "用近期已实现波动率分位数对Donchian突破强度加权，"
                "同样的价格突破在低波动环境获得更高权重，高波动环境"
                "打折，因为高波动下的突破更可能是噪音。"
            )
        )
    }


def build_alphas(df):
    alphas = {}
    alphas.update(build_volatility_regime(df))
    alphas.update(build_volatility_adjusted_breakout(df))
    return alphas
