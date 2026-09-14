"""独立"一月效应"OOS：做多IWM（小盘股）、做空SPY（大盘股），
只在每年1月持仓，其余11个月空仓。规则、成本、样本区间见
reports/mini_medallion_january_effect_oos/PREREGISTRATION.md。
和其余美元中性配对实验（月度再平衡、全年持仓）不同，这里的目标
仓位由"是否1月"这一日历条件门控，1月内部权重仍随价格自然漂移。
"""
import numpy as np
import pandas as pd


def validate_pair(small_cap, large_cap, maximum_gap_days=10):
    """对齐IWM/SPY调整收盘价，检查重复日期、非正值和异常缺口。"""
    small_cap = pd.Series(small_cap).dropna().astype(float).sort_index()
    large_cap = pd.Series(large_cap).dropna().astype(float).sort_index()
    if small_cap.empty or large_cap.empty:
        raise ValueError("IWM/SPY price series is empty")
    if small_cap.index.has_duplicates or large_cap.index.has_duplicates:
        raise ValueError("Duplicate January-effect price date")
    if (small_cap <= 0).any() or (large_cap <= 0).any():
        raise ValueError("January-effect adjusted prices must be positive")
    aligned = pd.concat(
        [small_cap.rename("IWM"), large_cap.rename("SPY")], axis=1, join="inner"
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
    large_cap,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """1月首个交易日建仓(leverage*0.5多IWM, leverage*0.5空SPY)，
    1月内部权重随价格自然漂移；2月首个交易日平仓归零；2-12月
    保持空仓，不产生任何持仓或成本。
    """
    aligned = validate_pair(small_cap, large_cap)
    returns = aligned.pct_change(fill_method=None).dropna()
    is_january = returns.index.month == 1
    is_first_january_day = is_january & ~np.roll(is_january, 1) if len(is_january) else is_january
    if len(is_january):
        is_first_january_day[0] = bool(is_january[0])
    is_first_february_day = (
        (returns.index.month == 2)
        & ~np.roll(returns.index.month == 2, 1)
    )
    if len(is_first_february_day):
        is_first_february_day[0] = False

    target_small_cap = leverage * 0.5
    target_large_cap = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_small_cap = np.empty(len(returns))
    weight_large_cap = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_small_cap_weight = 0.0
    prev_large_cap_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_first_january_day[i]:
            start_small_cap = target_small_cap
            start_large_cap = target_large_cap
        elif is_first_february_day[i]:
            start_small_cap = 0.0
            start_large_cap = 0.0
        elif is_january[i]:
            start_small_cap = prev_small_cap_weight
            start_large_cap = prev_large_cap_weight
        else:
            start_small_cap = 0.0
            start_large_cap = 0.0

        turnover[i] = (
            abs(start_small_cap - prev_small_cap_weight)
            + abs(start_large_cap - prev_large_cap_weight)
        )

        day_return = start_small_cap * row["IWM"] + start_large_cap * row["SPY"]
        financing = (
            (short_daily_financing + leverage_daily_financing)
            if is_january[i] else 0.0
        )
        cost = turnover[i] * leg_cost + financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        if is_january[i] and portfolio_growth != 0.0:
            weight_small_cap[i] = start_small_cap * (1.0 + row["IWM"]) / portfolio_growth
            weight_large_cap[i] = start_large_cap * (1.0 + row["SPY"]) / portfolio_growth
        else:
            weight_small_cap[i] = 0.0
            weight_large_cap[i] = 0.0
        prev_small_cap_weight = weight_small_cap[i]
        prev_large_cap_weight = weight_large_cap[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "small_cap_return": returns["IWM"],
        "large_cap_return": returns["SPY"],
        "weight_small_cap": weight_small_cap,
        "weight_large_cap": weight_large_cap,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
