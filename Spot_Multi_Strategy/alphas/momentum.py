"""A01 时序动量 与 A16 多周期动量集成。"""
from . import base

DEFAULT_MOMENTUM_HORIZONS = (12, 24, 48, 72, 120)
DEFAULT_VOL_WINDOW = 48


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

    return alphas
