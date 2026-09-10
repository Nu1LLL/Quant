"""扩展的组合层指标，补充metrics.py既有的calculate_metrics。

既有的metrics.py继续原样服务于strategies.py那条离散交易记录的回测路径，
完全不做改动。这里是给ensemble/risk_overlay这条"连续敞口"回测路径
准备的一组新指标（Sortino、Calmar、平均回撤、最长回撤天数、年化换手、
月度胜率等），输入是risk_overlay.apply_risk_overlay()那种逐根K线
net_pnl/position/turnover/cost的模拟结果。
"""
import numpy as np
import pandas as pd


def periods_per_year(open_time):
    open_time = pd.to_datetime(open_time, utc=True)
    if len(open_time) < 2:
        return 0.0
    bar_seconds = open_time.diff().dropna().median().total_seconds()
    if bar_seconds <= 0:
        return 0.0
    return 365.25 * 86400 / bar_seconds


def drawdown_series(equity):
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _drawdown_run_lengths(drawdown):
    lengths = []
    current = 0
    for value in drawdown:
        if value < 0:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def calendar_year_returns(open_time, returns):
    frame = pd.DataFrame({
        "open_time": pd.to_datetime(open_time, utc=True),
        "returns": returns.values
    })
    frame["year"] = frame["open_time"].dt.year
    yearly = frame.groupby("year")["returns"].apply(
        lambda values: float((1 + values).prod() - 1)
    )
    return yearly


def monthly_hit_rate(open_time, equity):
    frame = pd.DataFrame({
        "open_time": pd.to_datetime(open_time, utc=True),
        "equity": equity.values
    }).set_index("open_time")
    monthly_equity = frame["equity"].resample("ME").last().dropna()
    monthly_returns = monthly_equity.pct_change().dropna()
    if monthly_returns.empty:
        return 0.0
    return float((monthly_returns > 0).mean())


def calculate_extended_metrics(simulation_df, initial_capital=1.0):
    """simulation_df需要至少有open_time和net_pnl列；position/turnover/
    cost/rebalanced列如果存在会补充敞口和成本相关指标。
    """
    df = simulation_df.sort_values("open_time").reset_index(drop=True)
    returns = df["net_pnl"].fillna(0.0)
    equity = (1.0 + returns).cumprod()
    freq = periods_per_year(df["open_time"])

    elapsed_days = (
        (df["open_time"].iloc[-1] - df["open_time"].iloc[0]).total_seconds()
        / 86400
    )
    final_growth = float(equity.iloc[-1])

    if elapsed_days > 0 and final_growth > 0:
        cagr = final_growth ** (365.25 / elapsed_days) - 1
    else:
        cagr = 0.0

    return_std = returns.std(ddof=0)
    annualized_volatility = (
        float(return_std * np.sqrt(freq)) if freq > 0 else 0.0
    )

    sharpe_ratio = (
        float(returns.mean() / return_std * np.sqrt(freq))
        if return_std > 0 and freq > 0 else 0.0
    )

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std(ddof=0)
    sortino_ratio = (
        float(returns.mean() / downside_std * np.sqrt(freq))
        if downside_std and downside_std > 0 and freq > 0 else 0.0
    )

    drawdown = drawdown_series(equity)
    max_drawdown = float(drawdown.min())
    negative_drawdown = drawdown[drawdown < 0]
    average_drawdown = (
        float(negative_drawdown.mean()) if not negative_drawdown.empty else 0.0
    )

    days_per_bar = 365.25 / freq if freq > 0 else 0.0
    run_lengths = _drawdown_run_lengths(drawdown)
    longest_drawdown_days = (
        float(max(run_lengths) * days_per_bar) if run_lengths else 0.0
    )

    calmar_ratio = (
        float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    )

    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(-returns[returns < 0].sum())
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    if "position" in df.columns:
        in_market = df["position"] > 1e-9
        win_rate = (
            float((returns[in_market] > 0).mean())
            if in_market.any() else 0.0
        )
        average_exposure = float(df["position"].mean())
        exposure_pct = average_exposure
    else:
        win_rate = float((returns > 0).mean())
        average_exposure = np.nan
        exposure_pct = np.nan

    if "turnover" in df.columns:
        mean_turnover = float(df["turnover"].mean())
        annualized_turnover = float(mean_turnover * freq) if freq > 0 else 0.0
    else:
        mean_turnover = np.nan
        annualized_turnover = np.nan

    total_fees = (
        float((df["cost"] * initial_capital).sum())
        if "cost" in df.columns else np.nan
    )
    rebalance_count = (
        int(df["rebalanced"].sum()) if "rebalanced" in df.columns else np.nan
    )

    yearly_returns = calendar_year_returns(df["open_time"], returns)
    best_year = (
        float(yearly_returns.max()) if not yearly_returns.empty else 0.0
    )
    worst_year = (
        float(yearly_returns.min()) if not yearly_returns.empty else 0.0
    )

    return {
        "initial_capital": initial_capital,
        "final_value": initial_capital * final_growth,
        "total_return": final_growth - 1.0,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "average_drawdown": average_drawdown,
        "longest_drawdown_days": longest_drawdown_days,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "turnover_per_bar": mean_turnover,
        "annualized_turnover": annualized_turnover,
        "total_fees": total_fees,
        "rebalance_count": rebalance_count,
        "exposure_pct": exposure_pct,
        "average_exposure": average_exposure,
        "best_year": best_year,
        "worst_year": worst_year,
        "monthly_hit_rate": monthly_hit_rate(df["open_time"], equity),
        "elapsed_days": elapsed_days
    }
