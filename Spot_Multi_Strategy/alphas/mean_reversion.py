"""A04 短周期反转、A05 布林带反转强度、A06 RSI反转、A17 反转集成。"""
from indicators import calculate_rsi

from . import base

DEFAULT_REVERSION_HORIZONS = (1, 2, 3, 6)
DEFAULT_ZSCORE_WINDOW = 100


def build_short_term_mean_reversion(
    df,
    horizons=DEFAULT_REVERSION_HORIZONS,
    zscore_window=DEFAULT_ZSCORE_WINDOW
):
    close = df["close"]

    signals = {}
    for horizon in horizons:
        r_n = close / close.shift(horizon) - 1
        z = base.rolling_zscore(r_n, zscore_window)
        raw = -z
        normalized = base.squash(raw, scale=2.0)

        name = f"A04_short_reversion_{horizon}"
        signals[name] = base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="mean_reversion",
            lookback=horizon + zscore_window + 1,
            horizon=horizon,
            rationale=(
                f"{horizon}根K线收益率相对近期分布的z-score取负号，"
                "短期超跌/超涨后博弈均值回归，经济解释是短线过度反应"
                "会被后续流动性提供者部分纠正。"
            )
        )

    return signals


def build_bollinger_reversion_strength(df, window=20):
    close = df["close"]
    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std(ddof=0)
    distance = (close - rolling_mean) / rolling_std.replace(0, float("nan"))
    raw = -distance
    normalized = base.squash(raw, scale=2.0)

    name = f"A05_bollinger_reversion_{window}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="mean_reversion",
            lookback=window + 1,
            window=window,
            rationale=(
                "收盘价偏离布林带中轨的标准差距离取负号，是既有布林带"
                "策略布尔条件的连续版本：极端负偏离产生更强的正向"
                "回归alpha。"
            )
        )
    }


def build_rsi_reversion(df, window=14, scale=20.0):
    rsi = calculate_rsi(df["close"], window=window)
    raw = (50.0 - rsi) / scale

    name = f"A06_rsi_reversion_{window}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=raw,
            direction="mean_reversion",
            lookback=window + 1,
            window=window,
            scale=scale,
            rationale=(
                "把RSI变换为连续分数而非RSI<30的布尔条件，RSI低于50"
                "越多正向反转alpha越强，是一个透明的简化版本，"
                "不代表这个具体公式是最优的。"
            )
        )
    }


def build_alphas(df):
    short_reversion = build_short_term_mean_reversion(df)
    bollinger = build_bollinger_reversion_strength(df)
    rsi = build_rsi_reversion(df)

    ensemble_inputs = {}
    ensemble_inputs.update(short_reversion)
    ensemble_inputs.update(bollinger)
    ensemble_inputs.update(rsi)

    alphas = dict(ensemble_inputs)
    alphas["A17_mean_reversion_ensemble"] = base.ensemble_signal(
        name="A17_mean_reversion_ensemble",
        component_signals=ensemble_inputs,
        direction="mean_reversion",
        rationale=(
            "等权组合多个短周期反转信号（A04各周期+A05+A06），"
            "不挑选历史表现最好的单一反转信号，避免二次隐藏调参。"
        )
    )

    return alphas
