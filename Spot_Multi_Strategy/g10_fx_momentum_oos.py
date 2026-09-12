"""Lagged 12-1 relative momentum across three tradable currency trusts."""
import numpy as np
import pandas as pd


SYMBOLS = ("FXA", "FXB", "FXC")
LATEST_ACCEPTABLE_START = pd.Timestamp("2007-01-31", tz="UTC")


def validate_prices(prices):
    frame = pd.DataFrame(prices).copy().sort_index()
    if set(frame.columns) != set(SYMBOLS):
        raise ValueError(f"Expected exactly {SYMBOLS}")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame.loc[:, list(SYMBOLS)].dropna()
    if frame.empty or (frame <= 0).any().any() or frame.index.has_duplicates:
        raise ValueError("Currency price panel is empty, duplicated or nonpositive")
    gaps = frame.index.to_series().diff().dropna().dt.days
    if not gaps.empty and int(gaps.max()) > 10:
        raise ValueError("Currency common panel has a gap longer than ten days")
    return frame


def lagged_monthly_weights(prices):
    prices = validate_prices(prices)
    month_key = prices.index.tz_localize(None).to_period("M")
    month_end_rows = prices.groupby(month_key, sort=True).tail(1)
    scores = month_end_rows.shift(1).div(month_end_rows.shift(12)).sub(1.0)
    monthly_weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for timestamp, row in scores.dropna().iterrows():
        ordered = sorted(SYMBOLS, key=lambda symbol: (float(row[symbol]), symbol))
        monthly_weights.at[timestamp, ordered[0]] = -0.5
        monthly_weights.at[timestamp, ordered[-1]] = 0.5
    monthly_weights.loc[scores.isna().any(axis=1)] = np.nan
    daily_weights = monthly_weights.reindex(prices.index).ffill().shift(1)
    return daily_weights, scores


def run_scenario(
    prices, leverage, leg_cost=0.0005, annual_borrow=0.01,
    annual_financing=0.04,
):
    prices = validate_prices(prices)
    base_weights, scores = lagged_monthly_weights(prices)
    asset_returns = prices.pct_change(fill_method=None)
    weights = base_weights * leverage
    gross_return = (weights * asset_returns).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1, min_count=1)
    valid = weights.dropna(how="all")
    first_valid, last_valid = valid.index[0], valid.index[-1]
    turnover.loc[first_valid] = weights.loc[first_valid].abs().sum()
    maintenance_turnover = (weights * asset_returns).abs().sum(axis=1, min_count=1)
    trading_cost = (turnover + maintenance_turnover) * leg_cost
    trading_cost.loc[last_valid] += weights.loc[last_valid].abs().sum() * leg_cost
    short_borrow = leverage * 0.5 * annual_borrow / 252.0
    financing = max(leverage - 1.0, 0.0) * annual_financing / 252.0
    net_return = gross_return - trading_cost - short_borrow - financing
    detail = pd.DataFrame({
        "gross_return": gross_return,
        "turnover": turnover,
        "maintenance_turnover": maintenance_turnover,
        "trading_cost": trading_cost,
        "short_borrow": short_borrow,
        "financing_cost": financing,
        "net_return": net_return,
    }).dropna(subset=["net_return"])
    detail["equity"] = 10000.0 * (1.0 + detail["net_return"]).cumprod()
    return detail["net_return"], detail, weights.loc[detail.index], scores


def diagnostic_labels(prices, returns):
    basket = validate_prices(prices).mean(axis=1)
    foreign_currency_strong = basket.gt(basket.rolling(200, min_periods=200).mean()).shift(1)
    aligned = foreign_currency_strong.reindex(returns.index)
    return pd.DataFrame({
        "usd_regime": aligned.map({
            True: "foreign_fx_above_200d", False: "foreign_fx_below_200d"
        }),
        "calendar_year": returns.index.year.astype(str),
    }, index=returns.index)
