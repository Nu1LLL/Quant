"""独立动量因子OOS：做多MTUM（BlackRock，跟踪MSCI USA Momentum
Index，与AQR无关的独立指数方法论）、做空SPY，按月再平衡至美元
中性。规则、成本、样本区间见
reports/mini_medallion_momentum_factor_oos/PREREGISTRATION.md，
本模块只实现该文档已经锁定的规则，结构与quality_factor_oos.py
一致（同样是美元中性、月度再平衡的两腿组合），单独成文件以保持
本项目"一个假说一个文件"的既有约定，避免共享模块的列命名歧义。
"""
import numpy as np
import pandas as pd


def validate_pair(momentum, market, maximum_gap_days=10):
    """对齐MTUM/SPY调整收盘价，检查重复日期、非正值和异常缺口。"""
    momentum = pd.Series(momentum).dropna().astype(float).sort_index()
    market = pd.Series(market).dropna().astype(float).sort_index()
    if momentum.index.has_duplicates or market.index.has_duplicates:
        raise ValueError("Duplicate momentum-factor price date")
    if (momentum <= 0).any() or (market <= 0).any():
        raise ValueError("Momentum-factor adjusted prices must be positive")
    aligned = pd.concat(
        [momentum.rename("MTUM"), market.rename("SPY")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("MTUM and SPY have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long MTUM/SPY price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    momentum,
    market,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多MTUM, leverage*0.5空SPY)，日间权重
    随价格自然漂移，漂移用组合净收益（已扣成本）做归一化——分母
    必须是净收益，否则成本会被悄悄从权重漂移的分母里抵消掉。
    """
    aligned = validate_pair(momentum, market)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_momentum = leverage * 0.5
    target_market = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_momentum = np.empty(len(returns))
    weight_market = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_momentum_weight = 0.0
    prev_market_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_momentum = target_momentum
            start_market = target_market
            turnover[i] = (
                abs(target_momentum - prev_momentum_weight)
                + abs(target_market - prev_market_weight)
            )
        else:
            start_momentum = prev_momentum_weight
            start_market = prev_market_weight
            turnover[i] = 0.0

        day_return = start_momentum * row["MTUM"] + start_market * row["SPY"]
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_momentum[i] = start_momentum * (1.0 + row["MTUM"]) / portfolio_growth
        weight_market[i] = start_market * (1.0 + row["SPY"]) / portfolio_growth
        prev_momentum_weight = weight_momentum[i]
        prev_market_weight = weight_market[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "momentum_return": returns["MTUM"],
        "market_return": returns["SPY"],
        "weight_momentum": weight_momentum,
        "weight_market": weight_market,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
