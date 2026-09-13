"""Causal 12-1 cross-sectional momentum across ten country ETFs."""
import numpy as np
import pandas as pd


SYMBOLS = ("EWA", "EWC", "EWG", "EWH", "EWJ", "EWS", "EWU", "EWW", "EWQ", "EWP")
LATEST_ACCEPTABLE_START = pd.Timestamp("1996-04-30", tz="UTC")


def validate_prices(prices):
    frame = pd.DataFrame(prices).copy().sort_index()
    if set(frame.columns) != set(SYMBOLS):
        raise ValueError(f"Expected exactly {SYMBOLS}")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame.loc[:, list(SYMBOLS)].dropna()
    values = frame.to_numpy(dtype=float)
    if frame.empty or frame.index.has_duplicates or not np.isfinite(values).all():
        raise ValueError("Country panel is empty, duplicated or non-finite")
    if (values <= 0).any():
        raise ValueError("Country ETF prices must be positive")
    gaps = frame.index.to_series().diff().dropna().dt.days
    if not gaps.empty and int(gaps.max()) > 10:
        raise ValueError("Country common panel has a gap longer than ten days")
    return frame


def causal_weights(prices, skip=21, lookback=252):
    prices = validate_prices(prices)
    score = prices.shift(skip).div(prices.shift(lookback)).sub(1.0)
    targets = pd.DataFrame(np.nan, index=prices.index, columns=SYMBOLS)
    month = pd.Series(prices.index.year * 100 + prices.index.month, index=prices.index)
    month_end = month.ne(month.shift(-1))
    for timestamp in prices.index[month_end]:
        row = score.loc[timestamp]
        if row.notna().all():
            ordered = sorted(SYMBOLS, key=lambda symbol: (-row[symbol], symbol))
            target = pd.Series(0.0, index=SYMBOLS)
            target.loc[ordered[:3]] = 1.0 / 6.0
            target.loc[ordered[-3:]] = -1.0 / 6.0
            targets.loc[timestamp] = target
    targets = targets.ffill().fillna(0.0)
    return targets.shift(2).fillna(0.0), score, targets


def run_scenario(
    prices, leverage, leg_cost=0.0005, annual_borrow=0.01,
    annual_financing=0.04,
):
    prices = validate_prices(prices)
    base_weights, score, targets = causal_weights(prices)
    weights = base_weights * leverage
    asset_returns = prices.pct_change(fill_method=None)
    gross_return = (weights * asset_returns).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    maintenance_turnover = (weights * asset_returns).abs().sum(axis=1, min_count=1)
    trading_cost = (turnover + maintenance_turnover.fillna(0.0)) * leg_cost
    if weights.iloc[-1].abs().sum() > 0:
        trading_cost.iloc[-1] += weights.iloc[-1].abs().sum() * leg_cost
    short_notional = weights.clip(upper=0.0).abs().sum(axis=1)
    active = weights.abs().sum(axis=1).gt(0).astype(float)
    short_borrow = short_notional * annual_borrow / 252.0
    financing = active * max(leverage - 1.0, 0.0) * annual_financing / 252.0
    net_return = gross_return - trading_cost - short_borrow - financing
    detail = pd.DataFrame({
        "gross_return": gross_return, "turnover": turnover,
        "maintenance_turnover": maintenance_turnover, "trading_cost": trading_cost,
        "short_borrow": short_borrow, "financing_cost": financing,
        "net_return": net_return,
    }).dropna(subset=["net_return"])
    detail["equity"] = 10000.0 * (1.0 + detail["net_return"]).cumprod()
    return detail["net_return"], detail, weights.loc[detail.index], score, targets
