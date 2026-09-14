"""独立价值因子OOS：做多VTV（Vanguard，价值股）、做空VUG
（Vanguard，成长股），按月再平衡至美元中性。规则、成本、样本
区间见reports/mini_medallion_value_factor_oos/PREREGISTRATION.md，
本模块结构与quality_factor_oos.py/momentum_factor_oos.py/
size_factor_oos.py一致（同样是美元中性、月度再平衡的两腿组合），
单独成文件以保持本项目"一个假说一个文件"的既有约定。
"""
import numpy as np
import pandas as pd


def validate_pair(value_stocks, growth_stocks, maximum_gap_days=10):
    """对齐VTV/VUG调整收盘价，检查重复日期、非正值和异常缺口。"""
    value_stocks = pd.Series(value_stocks).dropna().astype(float).sort_index()
    growth_stocks = pd.Series(growth_stocks).dropna().astype(float).sort_index()
    if value_stocks.index.has_duplicates or growth_stocks.index.has_duplicates:
        raise ValueError("Duplicate value-factor price date")
    if (value_stocks <= 0).any() or (growth_stocks <= 0).any():
        raise ValueError("Value-factor adjusted prices must be positive")
    aligned = pd.concat(
        [value_stocks.rename("VTV"), growth_stocks.rename("VUG")],
        axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("VTV and VUG have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long VTV/VUG price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    value_stocks,
    growth_stocks,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多VTV, leverage*0.5空VUG)，日间权重
    随价格自然漂移，漂移用组合净收益（已扣成本）做归一化。
    """
    aligned = validate_pair(value_stocks, growth_stocks)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_value = leverage * 0.5
    target_growth = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_value = np.empty(len(returns))
    weight_growth = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_value_weight = 0.0
    prev_growth_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_value = target_value
            start_growth = target_growth
            turnover[i] = (
                abs(target_value - prev_value_weight)
                + abs(target_growth - prev_growth_weight)
            )
        else:
            start_value = prev_value_weight
            start_growth = prev_growth_weight
            turnover[i] = 0.0

        day_return = start_value * row["VTV"] + start_growth * row["VUG"]
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_value[i] = start_value * (1.0 + row["VTV"]) / portfolio_growth
        weight_growth[i] = start_growth * (1.0 + row["VUG"]) / portfolio_growth
        prev_value_weight = weight_value[i]
        prev_growth_weight = weight_growth[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "value_return": returns["VTV"],
        "growth_return": returns["VUG"],
        "weight_value": weight_value,
        "weight_growth": weight_growth,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
