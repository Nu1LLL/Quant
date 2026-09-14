"""独立通胀风险溢价OOS：做多TIP（通胀保值国债，BlackRock）、
做空IEF（名义中长期国债，BlackRock），按月再平衡至美元中性。
规则、成本、样本区间见
reports/mini_medallion_inflation_premium_oos/PREREGISTRATION.md，
本模块结构与term_premium_oos.py/credit_risk_premium_oos.py一致
（同样是美元中性、月度再平衡的两腿组合），单独成文件以保持本
项目"一个假说一个文件"的既有约定。
"""
import numpy as np
import pandas as pd


def validate_pair(inflation_protected, nominal, maximum_gap_days=10):
    """对齐TIP/IEF调整收盘价，检查重复日期、非正值和异常缺口。"""
    inflation_protected = pd.Series(inflation_protected).dropna().astype(float).sort_index()
    nominal = pd.Series(nominal).dropna().astype(float).sort_index()
    if inflation_protected.index.has_duplicates or nominal.index.has_duplicates:
        raise ValueError("Duplicate inflation-premium price date")
    if (inflation_protected <= 0).any() or (nominal <= 0).any():
        raise ValueError("Inflation-premium adjusted prices must be positive")
    aligned = pd.concat(
        [inflation_protected.rename("TIP"), nominal.rename("IEF")],
        axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("TIP and IEF have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long TIP/IEF price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    inflation_protected,
    nominal,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多TIP, leverage*0.5空IEF)，日间权重
    随价格自然漂移，漂移用组合净收益（已扣成本）做归一化。
    """
    aligned = validate_pair(inflation_protected, nominal)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_protected = leverage * 0.5
    target_nominal = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_protected = np.empty(len(returns))
    weight_nominal = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_protected_weight = 0.0
    prev_nominal_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_protected = target_protected
            start_nominal = target_nominal
            turnover[i] = (
                abs(target_protected - prev_protected_weight)
                + abs(target_nominal - prev_nominal_weight)
            )
        else:
            start_protected = prev_protected_weight
            start_nominal = prev_nominal_weight
            turnover[i] = 0.0

        day_return = start_protected * row["TIP"] + start_nominal * row["IEF"]
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_protected[i] = start_protected * (1.0 + row["TIP"]) / portfolio_growth
        weight_nominal[i] = start_nominal * (1.0 + row["IEF"]) / portfolio_growth
        prev_protected_weight = weight_protected[i]
        prev_nominal_weight = weight_nominal[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "inflation_protected_return": returns["TIP"],
        "nominal_return": returns["IEF"],
        "weight_protected": weight_protected,
        "weight_nominal": weight_nominal,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
