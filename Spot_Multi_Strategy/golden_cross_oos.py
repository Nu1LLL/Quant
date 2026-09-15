"""独立"黄金交叉/死亡交叉"趋势过滤OOS：50日均线上穿200日均线
做多SPY，下穿空仓。规则、成本、样本区间见
reports/mini_medallion_golden_cross_oos/PREREGISTRATION.md。和
halloween_seasonality_oos/turn_of_month_oos那两个纯日历信号不同，
均线关系依赖历史价格数据，因此目标仓位必须用前一天已确定的均线
关系（shift(1)），不能用当天才知道的收盘价，避免lookahead。
"""
import numpy as np
import pandas as pd

FAST_WINDOW = 50
SLOW_WINDOW = 200


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


def build_target_positions(prices, leverage):
    """第t天仓位由第t-1天收盘时50日/200日均线的相对关系决定
    （shift(1)），200日均线数据不足之前不产生任何仓位。
    """
    fast_ma = prices.rolling(FAST_WINDOW, min_periods=FAST_WINDOW).mean()
    slow_ma = prices.rolling(SLOW_WINDOW, min_periods=SLOW_WINDOW).mean()
    is_golden_cross = (fast_ma > slow_ma).shift(1)
    return is_golden_cross.fillna(False).astype(float) * leverage


def run_scenario(
    prices,
    leverage,
    leg_cost=0.0005,
    annual_leverage_financing=0.04,
):
    prices = validate_prices(prices)
    target_full = build_target_positions(prices, leverage)
    returns = prices.pct_change(fill_method=None).dropna()
    target = target_full.reindex(returns.index).fillna(0.0)

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
