"""A18 BTC领先/ETH滞后、A19 BTC与ETH相对强弱（pairwise版本）、
A20 市场广度、A23 跨资产相对强弱（N资产泛化版本）。

A18/A19需要btc_df与eth_df的open_time完全对齐（同一组时间戳、同一顺序），
调用方（alpha_research.py）负责在传入前用open_time做inner join对齐，
这里只做一次显式校验，绝不静默补齐或用未来数据填补缺口。

A20/A23是多资产版本，接受任意数量资产（不要求正好是BTC/ETH），
不同资产历史长度不一致时（比如SOL在Binance上市晚于BTC/ETH/BNB）
用outer join对齐，缺失的资产在那个时间点自然被排除出统计，不会
被静默填0或复用其他资产数据。
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


def _master_aligned_frame(asset_frames, series_builder):
    """把每个资产的某个时间序列对齐到共同的open_time网格（outer join）。

    某个资产在某个时间点还没有数据（比如SOL在2020年8月之前没有
    Binance现货记录）时，那个位置是NaN，参与后续跨资产统计时
    天然被pandas的skipna排除，不会被静默填成0或复用别的资产信息，
    不会因此产生虚假的市场广度或相对强弱读数。
    """
    series_by_symbol = {}
    for symbol, frame in asset_frames.items():
        open_time = pd.to_datetime(frame["open_time"], utc=True)
        series_by_symbol[symbol] = pd.Series(
            series_builder(frame).values, index=open_time
        )

    return pd.DataFrame(series_by_symbol).sort_index()


def build_market_breadth(asset_frames, horizon=24):
    """A20市场广度：同向动量资产占比，需要至少3个资产才有意义。

    经济解释：多个不完全同步的资产同时朝一个方向动，代表这是
    一次广泛的市场情绪驱动而不是单一资产的特异性波动，可能预示
    更强的趋势延续。架构上支持任意数量资产（当前研究用BTC/ETH/
    SOL/BNB四个），资产数量记录在metadata里。
    """
    momentum_frame = _master_aligned_frame(
        asset_frames,
        lambda frame: frame["close"] / frame["close"].shift(horizon) - 1
    )
    breadth_by_time = np.sign(momentum_frame).mean(axis=1, skipna=True)

    name = f"A20_market_breadth_{horizon}"
    signals = {}
    for symbol, frame in asset_frames.items():
        open_time = pd.to_datetime(frame["open_time"], utc=True)
        raw = breadth_by_time.reindex(open_time).reset_index(drop=True)
        signals[symbol] = {
            name: base.make_signal(
                name=name,
                raw=raw,
                normalized=raw.clip(-1, 1),
                direction="regime",
                lookback=horizon + 1,
                asset_count=len(asset_frames),
                rationale=(
                    f"{len(asset_frames)}个资产同向动量占比的市场广度，"
                    "多个资产同时朝一个方向动代表更广泛的市场情绪，"
                    "而不是单一资产的特异性波动。"
                )
            )
        }
    return signals


def build_cross_sectional_relative_strength(
    asset_frames,
    horizon=24,
    vol_window=48
):
    """A23：某资产的波动率标准化动量，相对"当时有数据的其余资产平均值"
    的差——是A19（只有BTC/ETH两个资产的pairwise版本）向N个资产的
    泛化，用于资产间相对配置倾斜，不代表做空。
    """
    normalized_momentum_frame = _master_aligned_frame(
        asset_frames,
        lambda frame: (
            (frame["close"] / frame["close"].shift(horizon) - 1)
            / base.realized_volatility(
                frame["close"], vol_window
            ).replace(0, np.nan)
        )
    )

    name = f"A23_cross_sectional_relative_strength_{horizon}"
    signals = {}
    for symbol, frame in asset_frames.items():
        peers = normalized_momentum_frame.drop(columns=[symbol])
        peer_average = peers.mean(axis=1, skipna=True)
        raw_by_time = normalized_momentum_frame[symbol] - peer_average

        open_time = pd.to_datetime(frame["open_time"], utc=True)
        raw = raw_by_time.reindex(open_time).reset_index(drop=True)
        normalized = base.squash(raw, scale=3.0)

        signals[symbol] = {
            name: base.make_signal(
                name=name,
                raw=raw,
                normalized=normalized,
                direction="relative_value",
                lookback=max(horizon, vol_window) + 1,
                horizon=horizon,
                peer_count=len(asset_frames) - 1,
                rationale=(
                    f"该资产波动率标准化动量相对其余{len(asset_frames) - 1}"
                    "个资产平均值的差，是A19两资产版本向多资产的泛化，"
                    "用于相对配置倾斜，不代表做空较弱资产。"
                )
            )
        }
    return signals


def build_alphas(btc_df, eth_df):
    alphas = {}
    alphas.update(build_btc_lead_eth_lag(btc_df, eth_df))
    alphas.update(build_relative_strength(btc_df, eth_df))
    return alphas
