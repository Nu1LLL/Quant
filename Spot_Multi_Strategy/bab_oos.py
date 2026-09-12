"""独立Betting-Against-Beta OOS：多头USMV（BlackRock）按1/滚动贝塔
放大到单位市场贝塔、空头SPHB（Invesco）按1/滚动贝塔缩小到单位
市场贝塔，SPY做贝塔估计基准，每月再平衡。规则、成本、样本区间见
reports/mini_medallion_bab_oos/PREREGISTRATION.md，本模块只实现
该文档已经锁定的规则。
"""
import numpy as np
import pandas as pd


def validate_triple(low_beta, high_beta, market, maximum_gap_days=10):
    """对齐USMV/SPHB/SPY调整收盘价，检查重复日期、非正值和异常缺口。"""
    series = {
        "low_beta": pd.Series(low_beta).dropna().astype(float).sort_index(),
        "high_beta": pd.Series(high_beta).dropna().astype(float).sort_index(),
        "market": pd.Series(market).dropna().astype(float).sort_index(),
    }
    for name, values in series.items():
        if values.index.has_duplicates:
            raise ValueError(f"Duplicate {name} price date")
        if (values <= 0).any():
            raise ValueError(f"{name} adjusted prices must be positive")
    aligned = pd.concat(
        [series["low_beta"].rename("LOW"), series["high_beta"].rename("HIGH"),
         series["market"].rename("MKT")],
        axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("LOW/HIGH/MKT have no overlapping dates")
    gaps = aligned.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(f"Abnormally long BAB price gap: {int(gaps.max())} days")
    return aligned


def causal_rolling_beta(asset_returns, market_returns, window=252):
    """252日滚动贝塔，用.shift(1)确保t日可用的贝塔只依赖截至t-1日
    收盘的数据，不包含t日本身——严格因果，不能在再平衡当天用到
    当天才知道的收益。"""
    covariance = asset_returns.rolling(window, min_periods=window).cov(market_returns)
    variance = market_returns.rolling(window, min_periods=window).var()
    beta = covariance / variance.replace(0.0, np.nan)
    return beta.shift(1)


def run_scenario(
    low_beta_prices,
    high_beta_prices,
    market_prices,
    leverage,
    leg_cost=0.0005,
    annual_short_financing=0.005,
    annual_leverage_financing=0.04,
    beta_window=252,
):
    aligned = validate_triple(low_beta_prices, high_beta_prices, market_prices)
    returns = aligned.pct_change(fill_method=None).dropna()

    beta_low = causal_rolling_beta(returns["LOW"], returns["MKT"], beta_window)
    beta_high = causal_rolling_beta(returns["HIGH"], returns["MKT"], beta_window)
    # 贝塔为0（数值上不可用）视为当天没有有效信号，不做任何仓位，
    # 不是对贝塔幅度做上下限截断
    beta_low = beta_low.replace(0.0, np.nan)
    beta_high = beta_high.replace(0.0, np.nan)

    month_keys = pd.Series(
        returns.index.year * 100 + returns.index.month, index=returns.index
    )
    is_new_month = month_keys.ne(month_keys.shift(1))
    is_new_month.iloc[0] = True

    short_daily_financing = annual_short_financing / 252.0
    leverage_daily_financing = annual_leverage_financing / 252.0

    weight_low = np.empty(len(returns))
    weight_high = np.empty(len(returns))
    turnover = np.empty(len(returns))
    net_return = np.empty(len(returns))
    gross_exposure = np.empty(len(returns))

    prev_low_weight = 0.0
    prev_high_weight = 0.0
    for i in range(len(returns)):
        row = returns.iloc[i]
        has_valid_beta = pd.notna(beta_low.iloc[i]) and pd.notna(beta_high.iloc[i])

        if is_new_month.iloc[i] and has_valid_beta:
            start_low = leverage * 0.5 / beta_low.iloc[i]
            start_high = -leverage * 0.5 / beta_high.iloc[i]
        elif is_new_month.iloc[i] and not has_valid_beta:
            # 没有有效贝塔之前，月初也保持空仓，不产生任何仓位
            start_low = 0.0
            start_high = 0.0
        else:
            start_low = prev_low_weight
            start_high = prev_high_weight

        turnover[i] = abs(start_low - prev_low_weight) + abs(start_high - prev_high_weight)
        short_notional = abs(start_high)
        gross = abs(start_low) + abs(start_high)
        gross_exposure[i] = gross

        day_return = start_low * row["LOW"] + start_high * row["HIGH"]
        financing = (
            short_notional * short_daily_financing
            + max(gross - 1.0, 0.0) * leverage_daily_financing
        )
        cost = turnover[i] * leg_cost + financing
        net_return[i] = day_return - cost

        portfolio_growth = 1.0 + net_return[i]
        weight_low[i] = start_low * (1.0 + row["LOW"]) / portfolio_growth
        weight_high[i] = start_high * (1.0 + row["HIGH"]) / portfolio_growth
        prev_low_weight = weight_low[i]
        prev_high_weight = weight_high[i]

    net = pd.Series(net_return, index=returns.index, name="net_return")
    detail = pd.DataFrame({
        "low_beta_return": returns["LOW"],
        "high_beta_return": returns["HIGH"],
        "market_return": returns["MKT"],
        "beta_low": beta_low.to_numpy(),
        "beta_high": beta_high.to_numpy(),
        "weight_low": weight_low,
        "weight_high": weight_high,
        "gross_exposure": gross_exposure,
        "turnover": turnover,
        "leverage": leverage,
        "net_return": net,
    })
    return net, detail
