"""A21/A22 资金费率alpha：第一个真正意义上"不是OHLCV数学变换"的数据源。

永续合约资金费率反映多空双方的拥挤程度（正费率=多头付钱给空头，
代表多头拥挤；负费率=空头拥挤），是仓位/情绪层面的信息，理论上
和价格本身的相关性较低，属于Mini-Medallion"多个低相关数据源"目标
里缺失的一块。数据来自futures_data.py已经在抓的Binance U本位永续
资金费率（8小时一次），不需要新的数据源或API密钥。

对齐规则：资金费率按8小时结算，现货K线是4小时一根，用
pd.merge_asof(direction="backward")把每根4小时K线对齐到"这根K线
开盘时刻已知的最近一次资金费率"，绝不使用尚未结算的未来费率。
"""
import numpy as np
import pandas as pd

from . import base

DEFAULT_ZSCORE_WINDOW = 90  # 90次结算 ≈ 30天
DEFAULT_TREND_WINDOW = 21   # 21次结算 ≈ 7天


def align_funding_rate_to_bars(df, funding_df):
    """把资金费率因果对齐到df的open_time网格，返回一个Series。

    funding_df需要有funding_time（已结算时刻）和funding_rate两列。
    对齐后第i行的值 = 在df.open_time[i]之前（含等于）最后一次已经
    结算的资金费率，如果这根K线开盘时还没有任何资金费率结算过，
    值是NaN（不会用未来费率回填）。
    """
    bars = pd.DataFrame({
        "open_time": pd.to_datetime(df["open_time"], utc=True)
    }).sort_values("open_time")

    funding = funding_df[["funding_time", "funding_rate"]].copy()
    funding["funding_time"] = pd.to_datetime(
        funding["funding_time"], utc=True
    )
    funding = funding.sort_values("funding_time")

    merged = pd.merge_asof(
        bars,
        funding,
        left_on="open_time",
        right_on="funding_time",
        direction="backward"
    )

    return merged["funding_rate"].reset_index(drop=True)


def build_funding_rate_crowding(
    df,
    funding_df,
    zscore_window=DEFAULT_ZSCORE_WINDOW
):
    """A21：资金费率相对近期分布的z-score取负号（反向/拥挤度alpha）。

    经济解释：资金费率异常偏高代表多头过度拥挤、需要持续付费维持
    仓位，历史上容易伴随挤仓式回调；异常偏低代表空头拥挤，容易
    伴随空头回补式反弹。这是一个有争议的假设（另一种可能是资金费率
    持续偏高只是反映强趋势会持续），本模块同时提供A22作为趋势延续
    版本，让实证结果决定哪个（如果有）成立，而不是预设立场。
    """
    funding_rate = align_funding_rate_to_bars(df, funding_df)
    z = base.rolling_zscore(funding_rate, zscore_window)
    raw = -z
    normalized = base.squash(raw, scale=2.0)

    name = f"A21_funding_rate_crowding_{zscore_window}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="mean_reversion",
            lookback=zscore_window + 1,
            data_source="binance_futures_funding_rate",
            zscore_window=zscore_window,
            rationale=(
                "资金费率相对近期分布的z-score取负号：费率异常偏高"
                "（多头拥挤）时博弈回调，异常偏低（空头拥挤）时博弈"
                "反弹。数据来自永续合约资金费率，不是价格的数学变换，"
                "理论上和OHLCV系alpha的相关性更低。"
            )
        )
    }


def build_funding_rate_trend(
    df,
    funding_df,
    trend_window=DEFAULT_TREND_WINDOW
):
    """A22：资金费率的滚动均值方向，作为趋势延续假设的对照版本。

    经济解释：资金费率持续为正代表市场持续看多情绪浓厚，可能反映
    真实的强趋势会延续，而不是过度拥挤即将反转。和A21的假设互斥，
    刻意保留两个方向相反的简单版本，交给alpha_research.py的IC分析
    判断哪个（如果有）在样本内成立。
    """
    funding_rate = align_funding_rate_to_bars(df, funding_df)
    raw = funding_rate.rolling(trend_window).mean()
    normalized = base.squash(raw, scale=0.0005)

    name = f"A22_funding_rate_trend_{trend_window}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="trend",
            lookback=trend_window + 1,
            data_source="binance_futures_funding_rate",
            trend_window=trend_window,
            rationale=(
                f"最近{trend_window}次资金费率结算的滚动均值，检验"
                "持续的资金费率方向是否代表趋势延续而不是拥挤反转——"
                "和A21的假设方向相反，两个alpha同时保留，不预设立场。"
            )
        )
    }


def build_alphas(df, funding_df):
    alphas = {}
    alphas.update(build_funding_rate_crowding(df, funding_df))
    alphas.update(build_funding_rate_trend(df, funding_df))
    return alphas
