"""BitMEX XBTUSD funding-history loader and delta-neutral carry model."""
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

import portfolio_metrics

FUNDING_URL = "https://www.bitmex.com/api/v1/funding"


def download_funding(
    symbol="XBTUSD", start_time="2016-01-01", end_time="2026-09-11",
    timeout=20, session=None
):
    request_session = session or requests.Session()
    cursor = pd.Timestamp(start_time, tz="UTC")
    end = pd.Timestamp(end_time, tz="UTC")
    records = []
    while cursor < end:
        response = request_session.get(
            FUNDING_URL,
            params={
                "symbol": symbol,
                "count": 500,
                "reverse": "false",
                "startTime": cursor.isoformat(),
                "endTime": end.isoformat(),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        records.extend(page)
        last = pd.Timestamp(page[-1]["timestamp"])
        next_cursor = last + timedelta(milliseconds=1)
        if next_cursor <= cursor:
            raise ValueError("BitMEX funding分页游标没有前进")
        cursor = next_cursor
        if len(page) < 500:
            break
    frame = pd.DataFrame(records)
    required = {"timestamp", "symbol", "fundingRate", "fundingInterval"}
    if not required.issubset(frame.columns):
        raise ValueError("BitMEX funding响应字段不符合预期")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    return (
        frame.dropna(subset=["timestamp", "fundingRate"])
        .drop_duplicates(subset=["timestamp", "symbol"])
        .sort_values("timestamp").reset_index(drop=True)
    )


def build_active_signal(funding_rate, policy):
    if policy == "ALWAYS":
        return pd.Series(1.0, index=funding_rate.index)
    if policy == "LAG_POSITIVE":
        return (funding_rate.shift(1) > 0.0).astype(float)
    raise ValueError(f"未知policy: {policy}")


def _summarize_daily(daily, initial_capital):
    metrics = portfolio_metrics.calculate_extended_metrics(
        daily, initial_capital=initial_capital
    )
    yearly = portfolio_metrics.calendar_year_returns(
        daily["open_time"], daily["net_pnl"]
    )
    counts = daily.assign(
        year=pd.to_datetime(daily["open_time"], utc=True).dt.year
    ).groupby("year").size()
    complete_years = yearly[counts >= 350]
    returns = daily.set_index("open_time")["net_pnl"]
    rolling_mean = returns.rolling(1095, min_periods=1095).mean()
    rolling_std = returns.rolling(1095, min_periods=1095).std(ddof=0)
    rolling_sharpe = (
        rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(365.25)
    )
    metrics.update({
        "positive_complete_year_ratio": (
            float((complete_years > 0).mean()) if len(complete_years) else 0.0
        ),
        "complete_year_count": int(len(complete_years)),
        "worst_rolling_3y_sharpe": (
            float(rolling_sharpe.min())
            if rolling_sharpe.notna().any() else np.nan
        ),
    })
    return yearly, metrics


def run_backtest(
    funding, policy, leverage, start="2016-09-12", end="2026-09-10",
    two_leg_cost=0.0015, annual_borrow=0.08, initial_capital=10000.0
):
    frame = funding.copy().sort_values("timestamp").reset_index(drop=True)
    frame["active"] = build_active_signal(frame["fundingRate"], policy)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    frame = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] < end_ts)]
    if frame.empty:
        raise ValueError("评价窗口没有funding记录")

    equity = float(initial_capital)
    notional = 0.0
    borrowed = 0.0
    previous_active = 0.0
    previous_month = None
    previous_timestamp = frame["timestamp"].iloc[0] - pd.Timedelta(hours=8)
    rows = []
    for row_number, row in frame.iterrows():
        timestamp = row["timestamp"]
        active = float(row["active"])
        month = (timestamp.year, timestamp.month)
        rebalance = (
            row_number == frame.index[0]
            or active != previous_active
            or month != previous_month
        )
        equity_before = equity
        turnover_dollars = 0.0
        if rebalance:
            target_notional = active * leverage * equity_before
            turnover_dollars = abs(target_notional - notional)
            equity -= turnover_dollars * two_leg_cost
            notional = target_notional
            borrowed = max(notional - equity_before, 0.0)
        elapsed_years = (
            (timestamp - previous_timestamp).total_seconds()
            / (365.25 * 86400.0)
        )
        funding_pnl = notional * float(row["fundingRate"])
        financing_cost = borrowed * annual_borrow * elapsed_years
        equity += funding_pnl - financing_cost
        trade_cost = turnover_dollars * two_leg_cost
        interval_return = equity / equity_before - 1.0
        rows.append({
            "timestamp": timestamp,
            "funding_rate": float(row["fundingRate"]),
            "active": active,
            "notional": notional,
            "turnover": turnover_dollars / equity_before,
            "trading_cost": trade_cost / equity_before,
            "financing_cost": financing_cost / equity_before,
            "net_pnl": interval_return,
            "equity": equity / initial_capital,
            "rebalanced": rebalance,
        })
        previous_active = active
        previous_month = month
        previous_timestamp = timestamp

    if notional > 0.0:
        equity_before = equity
        closing_cost = notional * two_leg_cost
        equity -= closing_cost
        rows[-1]["turnover"] += notional / equity_before
        rows[-1]["trading_cost"] += closing_cost / equity_before
        rows[-1]["net_pnl"] = (
            (1.0 + rows[-1]["net_pnl"])
            * (1.0 - closing_cost / equity_before) - 1.0
        )
        rows[-1]["equity"] = equity / initial_capital

    intervals = pd.DataFrame(rows)
    intervals["date"] = intervals["timestamp"].dt.normalize()
    daily = intervals.groupby("date", sort=True).agg({
        "net_pnl": lambda values: float((1.0 + values).prod() - 1.0),
        "active": "last",
        "notional": "last",
        "turnover": "sum",
        "trading_cost": "sum",
        "financing_cost": "sum",
        "equity": "last",
        "rebalanced": "max",
    }).reset_index().rename(columns={"date": "open_time"})
    daily["position"] = daily["active"] * leverage
    daily["cost"] = daily["trading_cost"] + daily["financing_cost"]
    yearly, metrics = _summarize_daily(daily, initial_capital)
    metrics.update({
        "total_trading_cost": float(
            intervals["trading_cost"].sum() * initial_capital
        ),
        "total_financing_cost": float(
            intervals["financing_cost"].sum() * initial_capital
        ),
        "active_interval_ratio": float(intervals["active"].mean()),
    })
    return intervals, daily, yearly, metrics
