"""独立发达市场（非美国）股票风险溢价OOS：做多EFA（BlackRock，
MSCI EAFE发达市场股票）、做空SPY，按月再平衡至美元中性。规则、
成本、样本区间见
reports/mini_medallion_developed_intl_equity_oos/PREREGISTRATION.md，
本模块结构与em_equity_oos.py/size_factor_oos.py等一致（同样是
美元中性、月度再平衡的两腿组合），单独成文件以保持本项目"一个
假说一个文件"的既有约定。
"""
import numpy as np
import pandas as pd


def validate_pair(international, domestic, maximum_gap_days=10):
    """对齐EFA/SPY调整收盘价，检查重复日期、非正值和异常缺口。"""
    international = pd.Series(international).dropna().astype(float).sort_index()
    domestic = pd.Series(domestic).dropna().astype(float).sort_index()
    if international.index.has_duplicates or domestic.index.has_duplicates:
        raise ValueError("Duplicate developed-intl-equity price date")
    if (international <= 0).any() or (domestic <= 0).any():
        raise ValueError("Developed-intl-equity adjusted prices must be positive")
    aligned = pd.concat(
        [international.rename("EFA"), domestic.rename("SPY")],
        axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("EFA and SPY have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long EFA/SPY price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    international,
    domestic,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多EFA, leverage*0.5空SPY)，日间权重
    随价格自然漂移，漂移用组合净收益（已扣成本）做归一化。
    """
    aligned = validate_pair(international, domestic)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_international = leverage * 0.5
    target_domestic = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_international = np.empty(len(returns))
    weight_domestic = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_international_weight = 0.0
    prev_domestic_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_international = target_international
            start_domestic = target_domestic
            turnover[i] = (
                abs(target_international - prev_international_weight)
                + abs(target_domestic - prev_domestic_weight)
            )
        else:
            start_international = prev_international_weight
            start_domestic = prev_domestic_weight
            turnover[i] = 0.0

        day_return = start_international * row["EFA"] + start_domestic * row["SPY"]
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_international[i] = (
            start_international * (1.0 + row["EFA"]) / portfolio_growth
        )
        weight_domestic[i] = start_domestic * (1.0 + row["SPY"]) / portfolio_growth
        prev_international_weight = weight_international[i]
        prev_domestic_weight = weight_domestic[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "international_return": returns["EFA"],
        "domestic_return": returns["SPY"],
        "weight_international": weight_international,
        "weight_domestic": weight_domestic,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
