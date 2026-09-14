"""独立日历季节性效应OOS："Sell in May and go away"万圣节指标：
11月-4月做多SPY、5月-10月持有现金。规则、成本、样本区间见
reports/mini_medallion_halloween_seasonality_oos/PREREGISTRATION.md。
和本项目其余基于价格/收益率的信号不同，这是纯日历规则，不依赖
任何需要延迟一天才能观察到的市场数据，因此目标仓位可以直接用
当天日历月份计算，不需要额外shift。
"""
import numpy as np
import pandas as pd

WINTER_MONTHS = {11, 12, 1, 2, 3, 4}


def validate_prices(prices, maximum_gap_days=10):
    prices = pd.Series(prices).dropna().astype(float).sort_index()
    if prices.empty:
        raise ValueError("SPY price series is empty")
    if prices.index.has_duplicates:
        raise ValueError("Duplicate SPY price date")
    if (prices <= 0).any():
        raise ValueError("SPY adjusted prices must be positive")
    gaps = prices.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(f"Abnormally long SPY price gap: {int(gaps.max())} days")
    return prices


def build_target_positions(index, leverage):
    """冬半年(11-4月)目标仓位=leverage，夏半年(5-10月)目标仓位=0，
    纯日历规则，直接由当天所属月份决定，不需要延迟。
    """
    months = index.month
    is_winter = np.isin(months, list(WINTER_MONTHS))
    return pd.Series(np.where(is_winter, leverage, 0.0), index=index)


def run_scenario(
    prices,
    leverage,
    leg_cost=0.0005,
    annual_leverage_financing=0.04,
):
    prices = validate_prices(prices)
    returns = prices.pct_change(fill_method=None).dropna()
    target = build_target_positions(returns.index, leverage)

    turnover = target.diff().abs()
    turnover.iloc[0] = target.iloc[0]
    trading_cost = turnover * leg_cost

    leverage_financing = (
        (target > 1.0).astype(float) * (leverage - 1.0)
        * annual_leverage_financing / 252.0
    )

    net_return = target * returns - trading_cost - leverage_financing
    net_return.name = "net_return"

    detail = pd.DataFrame({
        "spy_return": returns,
        "position": target,
        "turnover": turnover,
        "trading_cost": trading_cost,
        "leverage_financing": leverage_financing,
        "net_return": net_return,
    })
    return net_return, detail
