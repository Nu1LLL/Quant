"""独立信用风险溢价OOS：做多HYG（高收益公司债，BlackRock）、
做空LQD（投资级公司债，BlackRock），按月再平衡至美元中性。规则、
成本、样本区间见
reports/mini_medallion_credit_risk_premium_oos/PREREGISTRATION.md，
本模块结构与quality_factor_oos.py/momentum_factor_oos.py/
size_factor_oos.py一致（同样是美元中性、月度再平衡的两腿组合），
单独成文件以保持本项目"一个假说一个文件"的既有约定。
"""
import numpy as np
import pandas as pd


def validate_pair(high_yield, investment_grade, maximum_gap_days=10):
    """对齐HYG/LQD调整收盘价，检查重复日期、非正值和异常缺口。"""
    high_yield = pd.Series(high_yield).dropna().astype(float).sort_index()
    investment_grade = pd.Series(investment_grade).dropna().astype(float).sort_index()
    if high_yield.index.has_duplicates or investment_grade.index.has_duplicates:
        raise ValueError("Duplicate credit-premium price date")
    if (high_yield <= 0).any() or (investment_grade <= 0).any():
        raise ValueError("Credit-premium adjusted prices must be positive")
    aligned = pd.concat(
        [high_yield.rename("HYG"), investment_grade.rename("LQD")],
        axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("HYG and LQD have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long HYG/LQD price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    high_yield,
    investment_grade,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多HYG, leverage*0.5空LQD)，日间权重
    随价格自然漂移，漂移用组合净收益（已扣成本）做归一化。
    """
    aligned = validate_pair(high_yield, investment_grade)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_high_yield = leverage * 0.5
    target_investment_grade = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_high_yield = np.empty(len(returns))
    weight_investment_grade = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_high_yield_weight = 0.0
    prev_investment_grade_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_high_yield = target_high_yield
            start_investment_grade = target_investment_grade
            turnover[i] = (
                abs(target_high_yield - prev_high_yield_weight)
                + abs(target_investment_grade - prev_investment_grade_weight)
            )
        else:
            start_high_yield = prev_high_yield_weight
            start_investment_grade = prev_investment_grade_weight
            turnover[i] = 0.0

        day_return = (
            start_high_yield * row["HYG"] + start_investment_grade * row["LQD"]
        )
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_high_yield[i] = (
            start_high_yield * (1.0 + row["HYG"]) / portfolio_growth
        )
        weight_investment_grade[i] = (
            start_investment_grade * (1.0 + row["LQD"]) / portfolio_growth
        )
        prev_high_yield_weight = weight_high_yield[i]
        prev_investment_grade_weight = weight_investment_grade[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "high_yield_return": returns["HYG"],
        "investment_grade_return": returns["LQD"],
        "weight_high_yield": weight_high_yield,
        "weight_investment_grade": weight_investment_grade,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
