"""独立"月末月初效应"OOS：做多SPY，只在每月最后一个交易日+下月
前3个交易日（共4个交易日的窗口）持仓，其余交易日空仓。规则、
成本、样本区间见
reports/mini_medallion_turn_of_month_oos/PREREGISTRATION.md。
"""
import numpy as np
import pandas as pd


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


def build_window_flags(index):
    """用真实交易日历（不是自然日历）判定"月末月初窗口"：当月最后
    一个交易日，或者当月前3个交易日。用交易日在各自月份内的顺位
    （从头数、从尾数）判定，不依赖自然日历的月末/月初定义。
    """
    year_month = index.year * 100 + index.month
    frame = pd.DataFrame({"year_month": year_month}, index=index)
    trading_day_rank_from_start = frame.groupby("year_month").cumcount() + 1
    trading_day_rank_from_end = (
        frame.groupby("year_month").cumcount(ascending=False) + 1
    )
    is_last_trading_day = trading_day_rank_from_end.to_numpy() == 1
    is_first_three_trading_days = trading_day_rank_from_start.to_numpy() <= 3
    return is_last_trading_day | is_first_three_trading_days


def run_scenario(
    prices,
    leverage,
    leg_cost=0.0005,
    annual_leverage_financing=0.04,
):
    prices = validate_prices(prices)
    returns = prices.pct_change(fill_method=None).dropna()
    in_window = build_window_flags(returns.index)
    target = pd.Series(np.where(in_window, leverage, 0.0), index=returns.index)

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
        "in_window": in_window,
        "position": target,
        "turnover": turnover,
        "trading_cost": trading_cost,
        "leverage_financing": leverage_financing,
        "net_return": net_return,
    })
    return net_return, detail
