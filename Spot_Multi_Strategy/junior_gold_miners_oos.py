"""独立初级黄金矿业股相对黄金本身OOS：做多GDXJ（初级/小盘黄金
矿业股，VanEck）、做空GLD（黄金本身），按月再平衡至美元中性，
隔离初级矿业股相对金价本身的额外风险溢价。规则、成本、样本
区间见
reports/mini_medallion_junior_gold_miners_oos/PREREGISTRATION.md。
结构与gold_miners_oos.py（GDX/GLD）完全一致，单独成文件只是为了
避免复用那个模块里硬编码的"GDX"列名（这里的多头是GDXJ不是GDX），
保持本项目"一个假说一个文件"的既有约定。
"""
import numpy as np
import pandas as pd


def validate_pair(junior_miners, bullion, maximum_gap_days=10):
    """对齐GDXJ/GLD调整收盘价，检查重复日期、非正值和异常缺口。"""
    junior_miners = pd.Series(junior_miners).dropna().astype(float).sort_index()
    bullion = pd.Series(bullion).dropna().astype(float).sort_index()
    if junior_miners.index.has_duplicates or bullion.index.has_duplicates:
        raise ValueError("Duplicate junior-gold-miners price date")
    if (junior_miners <= 0).any() or (bullion <= 0).any():
        raise ValueError("Junior-gold-miners adjusted prices must be positive")
    aligned = pd.concat(
        [junior_miners.rename("GDXJ"), bullion.rename("GLD")],
        axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("GDXJ and GLD have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long GDXJ/GLD price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    junior_miners,
    bullion,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多GDXJ, leverage*0.5空GLD)，日间
    权重随价格自然漂移，漂移用组合净收益（已扣成本）做归一化。
    """
    aligned = validate_pair(junior_miners, bullion)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_miners = leverage * 0.5
    target_bullion = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_miners = np.empty(len(returns))
    weight_bullion = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_miners_weight = 0.0
    prev_bullion_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_miners = target_miners
            start_bullion = target_bullion
            turnover[i] = (
                abs(target_miners - prev_miners_weight)
                + abs(target_bullion - prev_bullion_weight)
            )
        else:
            start_miners = prev_miners_weight
            start_bullion = prev_bullion_weight
            turnover[i] = 0.0

        day_return = start_miners * row["GDXJ"] + start_bullion * row["GLD"]
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_miners[i] = start_miners * (1.0 + row["GDXJ"]) / portfolio_growth
        weight_bullion[i] = start_bullion * (1.0 + row["GLD"]) / portfolio_growth
        prev_miners_weight = weight_miners[i]
        prev_bullion_weight = weight_bullion[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "junior_miners_return": returns["GDXJ"],
        "bullion_return": returns["GLD"],
        "weight_miners": weight_miners,
        "weight_bullion": weight_bullion,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
