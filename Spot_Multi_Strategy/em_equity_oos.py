"""独立新兴市场股票风险溢价OOS：做多EEM（BlackRock，新兴市场
股票）、做空SPY，按月再平衡至美元中性。规则、成本、样本区间见
reports/mini_medallion_em_equity_oos/PREREGISTRATION.md，本模块
结构与quality_factor_oos.py/momentum_factor_oos.py/
size_factor_oos.py/credit_risk_premium_oos.py一致（同样是美元
中性、月度再平衡的两腿组合），单独成文件以保持本项目"一个假说
一个文件"的既有约定。
"""
import numpy as np
import pandas as pd


def validate_pair(emerging, developed, maximum_gap_days=10):
    """对齐EEM/SPY调整收盘价，检查重复日期、非正值和异常缺口。"""
    emerging = pd.Series(emerging).dropna().astype(float).sort_index()
    developed = pd.Series(developed).dropna().astype(float).sort_index()
    if emerging.index.has_duplicates or developed.index.has_duplicates:
        raise ValueError("Duplicate EM-equity price date")
    if (emerging <= 0).any() or (developed <= 0).any():
        raise ValueError("EM-equity adjusted prices must be positive")
    aligned = pd.concat(
        [emerging.rename("EEM"), developed.rename("SPY")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("EEM and SPY have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long EEM/SPY price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    emerging,
    developed,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多EEM, leverage*0.5空SPY)，日间权重
    随价格自然漂移，漂移用组合净收益（已扣成本）做归一化。
    """
    aligned = validate_pair(emerging, developed)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_emerging = leverage * 0.5
    target_developed = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_emerging = np.empty(len(returns))
    weight_developed = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_emerging_weight = 0.0
    prev_developed_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_emerging = target_emerging
            start_developed = target_developed
            turnover[i] = (
                abs(target_emerging - prev_emerging_weight)
                + abs(target_developed - prev_developed_weight)
            )
        else:
            start_emerging = prev_emerging_weight
            start_developed = prev_developed_weight
            turnover[i] = 0.0

        day_return = start_emerging * row["EEM"] + start_developed * row["SPY"]
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_emerging[i] = start_emerging * (1.0 + row["EEM"]) / portfolio_growth
        weight_developed[i] = start_developed * (1.0 + row["SPY"]) / portfolio_growth
        prev_emerging_weight = weight_emerging[i]
        prev_developed_weight = weight_developed[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "emerging_return": returns["EEM"],
        "developed_return": returns["SPY"],
        "weight_emerging": weight_emerging,
        "weight_developed": weight_developed,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
