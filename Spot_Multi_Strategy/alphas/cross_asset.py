"""A18 BTC领先/ETH滞后、A19 BTC与ETH相对强弱、A20 市场广度占位符。

A18/A19需要btc_df与eth_df的open_time完全对齐（同一组时间戳、同一顺序），
调用方（alpha_research.py）负责在传入前用open_time做inner join对齐，
这里只做一次显式校验，绝不静默补齐或用未来数据填补缺口。
"""
import numpy as np
import pandas as pd

from . import base

DEFAULT_LAGS = (1, 2, 3)


def _validate_aligned(btc_df, eth_df):
    if len(btc_df) != len(eth_df):
        raise ValueError("BTC与ETH数据长度不一致，无法计算跨资产alpha")

    if not btc_df["open_time"].reset_index(drop=True).equals(
        eth_df["open_time"].reset_index(drop=True)
    ):
        raise ValueError("BTC与ETH的open_time未对齐，无法计算跨资产alpha")


def build_btc_lead_eth_lag(btc_df, eth_df, lags=DEFAULT_LAGS):
    """用滞后lag根K线的BTC收益预测ETH未来收益。

    对齐规则：这个alpha在ETH的第i根K线（open_time[i]）用于交易时，
    只使用BTC在同一时间戳i及更早已经收盘的历史收益，不使用任何
    与ETH执行时刻同期或更晚的BTC信息。
    """
    _validate_aligned(btc_df, eth_df)

    btc_close = btc_df["close"].reset_index(drop=True)
    eth_index = eth_df.index

    signals = {}
    for lag in lags:
        btc_return = btc_close / btc_close.shift(lag) - 1
        raw = pd.Series(btc_return.values, index=eth_index)
        normalized = base.squash(raw, scale=0.03)

        name = f"A18_btc_lead_eth_lag_{lag}"
        signals[name] = base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="cross_asset",
            lookback=lag + 1,
            lag=lag,
            applicable_asset="ETHUSDT",
            rationale=(
                f"用滞后{lag}根K线的BTC收益预测ETH未来收益，只使用"
                "ETH执行时刻已知的BTC历史收益，不含任何同期未来信息。"
            )
        )

    return signals


def build_relative_strength(btc_df, eth_df, horizon=24, vol_window=48):
    _validate_aligned(btc_df, eth_df)

    btc_momentum = (
        btc_df["close"] / btc_df["close"].shift(horizon) - 1
    ).reset_index(drop=True)
    eth_momentum = (
        eth_df["close"] / eth_df["close"].shift(horizon) - 1
    ).reset_index(drop=True)

    btc_vol = base.realized_volatility(
        btc_df["close"], vol_window
    ).reset_index(drop=True)
    eth_vol = base.realized_volatility(
        eth_df["close"], vol_window
    ).reset_index(drop=True)

    btc_normalized_momentum = btc_momentum / btc_vol.replace(0, np.nan)
    eth_normalized_momentum = eth_momentum / eth_vol.replace(0, np.nan)

    raw = pd.Series(
        (btc_normalized_momentum - eth_normalized_momentum).values,
        index=eth_df.index
    )
    normalized = base.squash(raw, scale=3.0)

    name = f"A19_relative_strength_btc_eth_{horizon}"
    return {
        name: base.make_signal(
            name=name,
            raw=raw,
            normalized=normalized,
            direction="relative_value",
            lookback=max(horizon, vol_window) + 1,
            horizon=horizon,
            rationale=(
                "BTC与ETH经波动率标准化后的动量之差，用于调整两个"
                "资产之间的相对仓位配置（谁强多配一点），而不是"
                "自动做空较弱资产——现货组合没有做空能力。"
            )
        )
    }


def build_market_breadth_placeholder(asset_frames, horizon=24):
    """A20市场广度占位符：架构预留接口，不进入正式验收。

    asset_frames是{symbol: df}，用同向动量比例衡量市场广度。
    当前正式研究只稳定缓存了BTC/ETH两个资产，所以这个alpha默认
    research_only=True，不参与alpha_research.py的验收统计，
    只用于证明接口可以直接接受SOL/BNB等更多资产而不用改代码结构。
    """
    momentum_signs = []
    for symbol, frame in asset_frames.items():
        momentum = frame["close"] / frame["close"].shift(horizon) - 1
        momentum_signs.append(np.sign(momentum).rename(symbol))

    breadth = pd.concat(momentum_signs, axis=1).mean(axis=1)
    normalized = breadth.clip(-1, 1)

    name = f"A20_market_breadth_{horizon}"
    return {
        name: base.make_signal(
            name=name,
            raw=breadth,
            normalized=normalized,
            direction="regime",
            lookback=horizon + 1,
            asset_count=len(asset_frames),
            research_only=True,
            rationale=(
                "多资产同向动量比例的市场广度占位符，架构上支持"
                "后续加入SOL/BNB等更多资产；当前正式研究只用"
                "BTC/ETH两个资产，不作为验收信号。"
            )
        )
    }


def build_alphas(btc_df, eth_df):
    alphas = {}
    alphas.update(build_btc_lead_eth_lag(btc_df, eth_df))
    alphas.update(build_relative_strength(btc_df, eth_df))
    return alphas
