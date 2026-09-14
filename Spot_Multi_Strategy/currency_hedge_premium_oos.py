"""独立货币对冲溢价OOS：做多HEFA（对冲的发达市场股票，
BlackRock）、做空EFA（未对冲的发达市场股票），按月再平衡至美元
中性，隔离纯粹的货币对冲溢价。规则、成本、样本区间见
reports/mini_medallion_currency_hedge_premium_oos/PREREGISTRATION.md，
本模块结构与其他配对实验一致（同样是美元中性、月度再平衡的
两腿组合），单独成文件以保持本项目"一个假说一个文件"的既有约定。
"""
import numpy as np
import pandas as pd


def validate_pair(hedged, unhedged, maximum_gap_days=10):
    """对齐HEFA/EFA调整收盘价，检查重复日期、非正值和异常缺口。"""
    hedged = pd.Series(hedged).dropna().astype(float).sort_index()
    unhedged = pd.Series(unhedged).dropna().astype(float).sort_index()
    if hedged.empty or unhedged.empty:
        raise ValueError("HEFA/EFA price series is empty")
    if hedged.index.has_duplicates or unhedged.index.has_duplicates:
        raise ValueError("Duplicate currency-hedge price date")
    if (hedged <= 0).any() or (unhedged <= 0).any():
        raise ValueError("Currency-hedge adjusted prices must be positive")
    aligned = pd.concat(
        [hedged.rename("HEFA"), unhedged.rename("EFA")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("HEFA and EFA have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long HEFA/EFA price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    hedged,
    unhedged,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多HEFA, leverage*0.5空EFA)，日间
    权重随价格自然漂移，漂移用组合净收益（已扣成本）做归一化。
    """
    aligned = validate_pair(hedged, unhedged)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_hedged = leverage * 0.5
    target_unhedged = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_hedged = np.empty(len(returns))
    weight_unhedged = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_hedged_weight = 0.0
    prev_unhedged_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_hedged = target_hedged
            start_unhedged = target_unhedged
            turnover[i] = (
                abs(target_hedged - prev_hedged_weight)
                + abs(target_unhedged - prev_unhedged_weight)
            )
        else:
            start_hedged = prev_hedged_weight
            start_unhedged = prev_unhedged_weight
            turnover[i] = 0.0

        day_return = start_hedged * row["HEFA"] + start_unhedged * row["EFA"]
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_hedged[i] = start_hedged * (1.0 + row["HEFA"]) / portfolio_growth
        weight_unhedged[i] = start_unhedged * (1.0 + row["EFA"]) / portfolio_growth
        prev_hedged_weight = weight_hedged[i]
        prev_unhedged_weight = weight_unhedged[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "hedged_return": returns["HEFA"],
        "unhedged_return": returns["EFA"],
        "weight_hedged": weight_hedged,
        "weight_unhedged": weight_unhedged,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
