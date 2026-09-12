"""独立久期/期限溢价OOS：做多TLT（20年以上美国国债，BlackRock）、
做空SHY（1-3年美国国债，BlackRock），按月再平衡至美元中性。规则、
成本、样本区间见
reports/mini_medallion_term_premium_oos/PREREGISTRATION.md，本
模块结构与credit_risk_premium_oos.py等一致（同样是美元中性、
月度再平衡的两腿组合），单独成文件以保持本项目"一个假说一个
文件"的既有约定。
"""
import numpy as np
import pandas as pd


def validate_pair(long_duration, short_duration, maximum_gap_days=10):
    """对齐TLT/SHY调整收盘价，检查重复日期、非正值和异常缺口。"""
    long_duration = pd.Series(long_duration).dropna().astype(float).sort_index()
    short_duration = pd.Series(short_duration).dropna().astype(float).sort_index()
    if long_duration.index.has_duplicates or short_duration.index.has_duplicates:
        raise ValueError("Duplicate term-premium price date")
    if (long_duration <= 0).any() or (short_duration <= 0).any():
        raise ValueError("Term-premium adjusted prices must be positive")
    aligned = pd.concat(
        [long_duration.rename("TLT"), short_duration.rename("SHY")],
        axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("TLT and SHY have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long TLT/SHY price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    long_duration,
    short_duration,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多TLT, leverage*0.5空SHY)，日间权重
    随价格自然漂移，漂移用组合净收益（已扣成本）做归一化。
    """
    aligned = validate_pair(long_duration, short_duration)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_long = leverage * 0.5
    target_short = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_long = np.empty(len(returns))
    weight_short = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_long_weight = 0.0
    prev_short_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_long = target_long
            start_short = target_short
            turnover[i] = (
                abs(target_long - prev_long_weight)
                + abs(target_short - prev_short_weight)
            )
        else:
            start_long = prev_long_weight
            start_short = prev_short_weight
            turnover[i] = 0.0

        day_return = start_long * row["TLT"] + start_short * row["SHY"]
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_long[i] = start_long * (1.0 + row["TLT"]) / portfolio_growth
        weight_short[i] = start_short * (1.0 + row["SHY"]) / portfolio_growth
        prev_long_weight = weight_long[i]
        prev_short_weight = weight_short[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "long_duration_return": returns["TLT"],
        "short_duration_return": returns["SHY"],
        "weight_long": weight_long,
        "weight_short": weight_short,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
