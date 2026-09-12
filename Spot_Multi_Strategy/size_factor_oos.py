"""独立规模因子OOS：做多IWM（BlackRock，跟踪罗素2000小盘股指数）、
做空SPY，按月再平衡至美元中性。规则、成本、样本区间见
reports/mini_medallion_size_factor_oos/PREREGISTRATION.md，本模块
结构与quality_factor_oos.py/momentum_factor_oos.py一致（同样是
美元中性、月度再平衡的两腿组合），单独成文件以保持本项目"一个
假说一个文件"的既有约定。
"""
import numpy as np
import pandas as pd


def validate_pair(small_cap, market, maximum_gap_days=10):
    """对齐IWM/SPY调整收盘价，检查重复日期、非正值和异常缺口。"""
    small_cap = pd.Series(small_cap).dropna().astype(float).sort_index()
    market = pd.Series(market).dropna().astype(float).sort_index()
    if small_cap.index.has_duplicates or market.index.has_duplicates:
        raise ValueError("Duplicate size-factor price date")
    if (small_cap <= 0).any() or (market <= 0).any():
        raise ValueError("Size-factor adjusted prices must be positive")
    aligned = pd.concat(
        [small_cap.rename("IWM"), market.rename("SPY")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("IWM and SPY have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long IWM/SPY price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    small_cap,
    market,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多IWM, leverage*0.5空SPY)，日间权重
    随价格自然漂移，漂移用组合净收益（已扣成本）做归一化。
    """
    aligned = validate_pair(small_cap, market)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_small_cap = leverage * 0.5
    target_market = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_small_cap = np.empty(len(returns))
    weight_market = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_small_cap_weight = 0.0
    prev_market_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_small_cap = target_small_cap
            start_market = target_market
            turnover[i] = (
                abs(target_small_cap - prev_small_cap_weight)
                + abs(target_market - prev_market_weight)
            )
        else:
            start_small_cap = prev_small_cap_weight
            start_market = prev_market_weight
            turnover[i] = 0.0

        day_return = start_small_cap * row["IWM"] + start_market * row["SPY"]
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_small_cap[i] = start_small_cap * (1.0 + row["IWM"]) / portfolio_growth
        weight_market[i] = start_market * (1.0 + row["SPY"]) / portfolio_growth
        prev_small_cap_weight = weight_small_cap[i]
        prev_market_weight = weight_market[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "small_cap_return": returns["IWM"],
        "market_return": returns["SPY"],
        "weight_small_cap": weight_small_cap,
        "weight_market": weight_market,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
