"""A01时序动量、A16多周期集成、A24跳过期动量、A25符号化TSMOM。

A24/A25不是自己拍脑袋设计的公式，是文献/业界已经发表过的成熟因子
定义的直接实现（用于加密货币这份数据，不是针对这份数据拟合出来的）：

- A24跳过期动量：Jegadeesh & Titman (1993)"Returns to Buying Winners
  and Selling Losers"确立的标准动量构造——用t-N到t-skip的收益率，
  跳过最近skip根K线，避免短期反转污染动量信号。
- A25符号化、波动率目标化时序动量：Moskowitz, Ooi & Pedersen (2012)
  "Time Series Momentum"（AQR）——用sign(过去收益率)而不是收益率的
  具体幅度，理由是幅度本身噪声很大，符号更稳定；原论文里波动率
  目标化发生在仓位构建阶段（本项目对应risk_overlay.py的组合层
  波动率目标），这里alpha只产生方向信号本身。
"""
import numpy as np
import pandas as pd

from . import base

DEFAULT_MOMENTUM_HORIZONS = (12, 24, 48, 72, 120)
DEFAULT_VOL_WINDOW = 48
DEFAULT_SKIP_MOMENTUM_HORIZONS = (72, 120, 168)
DEFAULT_SKIP_BARS = 6
DEFAULT_TSMOM_HORIZONS = (24, 72, 168)


def build_time_series_momentum(
    df,
    horizons=DEFAULT_MOMENTUM_HORIZONS,
    vol_window=DEFAULT_VOL_WINDOW
):
    close = df["close"]
    vol = base.realized_volatility(close, vol_window)

    signals = {}
    for horizon in horizons:
        raw_momentum = close / close.shift(horizon) - 1
        vol_adjusted = raw_momentum / vol.replace(0, float("nan"))
        normalized = base.squash(vol_adjusted, scale=3.0)

        name = f"A01_ts_momentum_{horizon}"
        signals[name] = base.make_signal(
            name=name,
            raw=raw_momentum,
            normalized=normalized,
            direction="trend",
            lookback=max(horizon, vol_window) + 1,
            horizon=horizon,
            rationale=(
                f"{horizon}根K线收益率经近期已实现波动率标准化，"
                "衡量趋势延续强度。经济解释：动量效应在加密货币"
                "现货历史上长期存在（资金持续流入/流出驱动的趋势延续）。"
            )
        )

    return signals


def build_skip_period_momentum(
    df,
    horizons=DEFAULT_SKIP_MOMENTUM_HORIZONS,
    skip=DEFAULT_SKIP_BARS,
    vol_window=DEFAULT_VOL_WINDOW
):
    """A24：Jegadeesh & Titman标准动量构造——从t-skip往前数horizon根
    K线的收益率，跳过最近skip根K线避免短期反转污染。
    """
    close = df["close"]
    vol = base.realized_volatility(close, vol_window)

    signals = {}
    for horizon in horizons:
        raw_momentum = (
            close.shift(skip) / close.shift(skip + horizon) - 1
        )
        vol_adjusted = raw_momentum / vol.replace(0, float("nan"))
        normalized = base.squash(vol_adjusted, scale=3.0)

        name = f"A24_skip_momentum_{horizon}_{skip}"
        signals[name] = base.make_signal(
            name=name,
            raw=raw_momentum,
            normalized=normalized,
            direction="trend",
            lookback=max(horizon + skip, vol_window) + 1,
            horizon=horizon,
            skip=skip,
            literature_reference="Jegadeesh & Titman (1993)",
            rationale=(
                f"从t-{skip}往前数{horizon}根K线的收益率（跳过最近"
                f"{skip}根K线），标准的12-1式动量构造，避免短期反转"
                "污染动量信号——不是针对这份数据拟合出来的公式。"
            )
        )

    return signals


def build_sign_based_tsmom(df, horizons=DEFAULT_TSMOM_HORIZONS):
    """A25：Moskowitz, Ooi & Pedersen (2012)时序动量——用符号而不是
    幅度，理由是收益率幅度噪声很大，方向本身更稳定。原论文的波动率
    目标化发生在仓位构建阶段，这里只产生方向信号。
    """
    close = df["close"]

    signals = {}
    for horizon in horizons:
        raw_momentum = close / close.shift(horizon) - 1
        sign_signal = pd.Series(
            np.sign(raw_momentum.to_numpy()), index=raw_momentum.index
        )

        name = f"A25_sign_tsmom_{horizon}"
        signals[name] = base.make_signal(
            name=name,
            raw=sign_signal,
            normalized=sign_signal,
            direction="trend",
            lookback=horizon + 1,
            horizon=horizon,
            literature_reference="Moskowitz, Ooi & Pedersen (2012)",
            rationale=(
                f"过去{horizon}根K线收益率的符号（不是幅度），"
                "AQR《Time Series Momentum》确立的构造方式，理由是"
                "幅度本身噪声很大、符号更稳定——不是针对这份数据"
                "拟合出来的公式。"
            )
        )

    return signals


def build_alphas(df):
    momentum_signals = build_time_series_momentum(df)

    alphas = dict(momentum_signals)
    alphas["A16_momentum_ensemble"] = base.ensemble_signal(
        name="A16_momentum_ensemble",
        component_signals=momentum_signals,
        direction="trend",
        rationale=(
            "等权组合A01各周期动量信号，不在alpha候选库内部挑选"
            "历史表现最好的单一周期，避免二次隐藏调参。"
        )
    )
    alphas.update(build_skip_period_momentum(df))
    alphas.update(build_sign_based_tsmom(df))

    return alphas
