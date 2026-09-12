"""独立质量因子OOS：做多QUAL（BlackRock，与AQR无关的独立发行方）、
做空SPY，按月再平衡至美元中性。规则、成本、样本区间见
reports/mini_medallion_quality_factor_oos/PREREGISTRATION.md，
本模块只负责实现该文档已经锁定的规则，不做任何事后调整。
"""
import numpy as np
import pandas as pd


def validate_pair(quality, market, maximum_gap_days=10):
    """对齐QUAL/SPY调整收盘价，检查重复日期、非正值和异常缺口。"""
    quality = pd.Series(quality).dropna().astype(float).sort_index()
    market = pd.Series(market).dropna().astype(float).sort_index()
    if quality.index.has_duplicates or market.index.has_duplicates:
        raise ValueError("Duplicate quality-factor price date")
    if (quality <= 0).any() or (market <= 0).any():
        raise ValueError("Quality-factor adjusted prices must be positive")
    aligned = pd.concat(
        [quality.rename("QUAL"), market.rename("SPY")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("QUAL and SPY have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(
            f"Abnormally long QUAL/SPY price gap: {int(gaps.max())} days"
        )
    return aligned


def run_scenario(
    quality,
    market,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.003,
    annual_leverage_financing=0.04,
):
    """月度再平衡至(leverage*0.5多QUAL, leverage*0.5空SPY)，日间权重
    随价格自然漂移（不是每天重置），漂移用组合价值变化做归一化。

    成本包含：每次再平衡的换手×leg_cost（双边各5bp）、SPY空头持仓的
    通用抵押借券成本（年化，按252个交易日均摊）、超过1x部分的杠杆
    融资成本（与本项目QAI/DBV实验同一口径，年化4%）。
    """
    aligned = validate_pair(quality, market)
    returns = aligned.pct_change(fill_method=None).dropna()
    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_rebalance_day = month_keys.ne(month_keys.shift(1))
    is_rebalance_day.iloc[0] = True

    target_quality = leverage * 0.5
    target_market = -leverage * 0.5
    short_daily_financing = leverage * 0.5 * annual_short_financing / 252.0
    leverage_daily_financing = max(leverage - 1.0, 0.0) * annual_leverage_financing / 252.0

    weight_quality = np.empty(len(returns))
    weight_market = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))

    prev_quality_weight = 0.0
    prev_market_weight = 0.0
    for i, (_, row) in enumerate(returns.iterrows()):
        if is_rebalance_day.iloc[i]:
            start_quality = target_quality
            start_market = target_market
            turnover[i] = (
                abs(target_quality - prev_quality_weight)
                + abs(target_market - prev_market_weight)
            )
        else:
            start_quality = prev_quality_weight
            start_market = prev_market_weight
            turnover[i] = 0.0

        day_return = start_quality * row["QUAL"] + start_market * row["SPY"]
        cost = turnover[i] * leg_cost + short_daily_financing + leverage_daily_financing
        net_return[i] = day_return - cost

        # 权重漂移的分母必须是净收益（组合总值的实际变化，成本已经
        # 从组合现金里扣掉），不能用未扣成本的day_return，否则会把
        # 成本悄悄从分母里"抵消"掉，长期累积会让权重系统性偏高。
        portfolio_growth = 1.0 + net_return[i]
        weight_quality[i] = start_quality * (1.0 + row["QUAL"]) / portfolio_growth
        weight_market[i] = start_market * (1.0 + row["SPY"]) / portfolio_growth
        prev_quality_weight = weight_quality[i]
        prev_market_weight = weight_market[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "quality_return": returns["QUAL"],
        "market_return": returns["SPY"],
        "weight_quality": weight_quality,
        "weight_market": weight_market,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
