"""OHLCV alpha信号库。

这是与strategies.py并行的新alpha生成层，strategies.py（趋势突破/
趋势回调/震荡均值回归三个既有策略）不受影响，继续按原样工作。
本模块只负责产生连续的alpha分数，不直接下单，也没有替换现有的
sleeve执行引擎（engine.py）。
"""
from . import (
    base,
    cross_asset,
    mean_reversion,
    momentum,
    trend,
    volatility,
    volume
)
from .base import AlphaSignal

SINGLE_ASSET_MODULES = (momentum, trend, mean_reversion, volatility, volume)


def build_single_asset_alphas(df):
    """返回只依赖单一资产OHLCV数据的alpha集合（A01-A17）。"""
    df = df.reset_index(drop=True)

    alphas = {}
    for module in SINGLE_ASSET_MODULES:
        alphas.update(module.build_alphas(df))

    return alphas


def build_cross_asset_alphas(btc_df, eth_df):
    """返回需要BTC与ETH同时对齐的跨资产alpha（A18-A19）。"""
    return cross_asset.build_alphas(
        btc_df.reset_index(drop=True),
        eth_df.reset_index(drop=True)
    )


def build_alpha_library(df, btc_df=None, eth_df=None):
    """返回某个资产的完整alpha集合。

    只有当btc_df和eth_df都提供时才附加跨资产alpha（A18/A19），
    这两者通常只在为ETHUSDT构建alpha库时提供，因为A18/A19的设计
    用途是用BTC信息预测/调整ETH仓位。
    """
    alphas = build_single_asset_alphas(df)

    if btc_df is not None and eth_df is not None:
        alphas.update(build_cross_asset_alphas(btc_df, eth_df))

    return alphas


__all__ = [
    "AlphaSignal",
    "base",
    "cross_asset",
    "mean_reversion",
    "momentum",
    "trend",
    "volatility",
    "volume",
    "build_single_asset_alphas",
    "build_cross_asset_alphas",
    "build_alpha_library"
]
