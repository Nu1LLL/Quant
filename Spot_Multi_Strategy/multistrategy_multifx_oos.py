"""Preregistered PHDG plus six-currency, two-strategy portfolio."""
import numpy as np
import pandas as pd


PHDG = "PHDG"
FX_SYMBOLS = ("6E=F", "6J=F", "6B=F", "6A=F", "6C=F", "6S=F")
SYMBOLS = (PHDG,) + FX_SYMBOLS
LATEST_ACCEPTABLE_START = pd.Timestamp("2013-01-31", tz="UTC")


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
        raise ValueError("Multistrategy panel is empty, duplicated or non-finite")
    if (values <= 0).any():
        raise ValueError("Multistrategy prices must be positive")
    gaps = frame.index.to_series().diff().dropna().dt.days
    if not gaps.empty and int(gaps.max()) > 10:
        raise ValueError("Multistrategy common panel has a gap longer than ten days")
    return frame


def _capped_inverse_vol(volatility, signs, cap=0.30):
    vol = pd.Series(volatility, index=FX_SYMBOLS, dtype=float)
    direction = pd.Series(signs, index=FX_SYMBOLS, dtype=float)
    if vol.isna().any() or direction.isna().any() or (vol <= 0).any():
        return pd.Series(np.nan, index=FX_SYMBOLS, dtype=float)
    inverse = 1.0 / vol
    absolute = pd.Series(0.0, index=FX_SYMBOLS)
    remaining = 1.0
    eligible = list(FX_SYMBOLS)
    for _ in range(10):
        if not eligible or remaining <= 1e-12:
            break
        denominator = inverse.loc[eligible].sum()
        proposal = remaining * inverse.loc[eligible] / denominator
        capped = [symbol for symbol in eligible if proposal[symbol] > cap]
        if not capped:
            absolute.loc[eligible] = proposal
            remaining = 0.0
            break
        for symbol in capped:
            absolute[symbol] = cap
            remaining -= cap
            eligible.remove(symbol)
    if remaining > 1e-9:
        raise ValueError("Inverse-vol cap could not allocate full FX trend sleeve")
    return absolute * direction


def causal_sleeve_targets(prices):
    prices = validate_prices(prices)
    fx = prices.loc[:, FX_SYMBOLS]
    returns = fx.pct_change(fill_method=None)
    trend_momentum = fx / fx.shift(252) - 1.0
    trend_sign = trend_momentum.gt(0).astype(float) * 2.0 - 1.0
    trend_sign = trend_sign.where(trend_momentum.notna())
    volatility = returns.rolling(63, min_periods=63).std() * np.sqrt(252.0)
    cross_momentum = fx.shift(21) / fx.shift(252) - 1.0
    month = pd.Series(prices.index.year * 100 + prices.index.month, index=prices.index)
    month_end = month.ne(month.shift(-1))

    trend = pd.DataFrame(np.nan, index=prices.index, columns=FX_SYMBOLS)
    cross = pd.DataFrame(np.nan, index=prices.index, columns=FX_SYMBOLS)
    for date in prices.index[month_end]:
        if trend_sign.loc[date].notna().all() and volatility.loc[date].notna().all():
            trend.loc[date] = _capped_inverse_vol(volatility.loc[date], trend_sign.loc[date])
        if cross_momentum.loc[date].notna().all():
            ordered = sorted(FX_SYMBOLS, key=lambda symbol: (cross_momentum.loc[date, symbol], symbol))
            row = pd.Series(0.0, index=FX_SYMBOLS)
            row.loc[ordered[:2]] = -0.25
            row.loc[ordered[-2:]] = 0.25
            cross.loc[date] = row
    trend = trend.ffill().fillna(0.0)
    cross = cross.ffill().fillna(0.0)
    return trend, cross, volatility, trend_momentum, cross_momentum


def apply_bankruptcy(raw_returns):
    reported = pd.Series(raw_returns).copy()
    bankrupt = reported.le(-1.0)
    if bankrupt.any():
        first = bankrupt[bankrupt].index[0]
        reported.loc[first] = -1.0
        reported.loc[reported.index > first] = 0.0
    return reported


def run_scenario(
    prices, leverage, phdg_cost=0.001, fx_cost=0.0005,
    annual_roll_haircut=0.01, annual_financing=0.04,
):
    prices = validate_prices(prices)
    trend_targets, cross_targets, volatility, trend_momentum, cross_momentum = causal_sleeve_targets(prices)
    phdg_target = pd.Series(0.40, index=prices.index)
    phdg_position = phdg_target.shift(2).fillna(0.0) * leverage
    trend_position = trend_targets.shift(2).fillna(0.0) * 0.30 * leverage
    cross_position = cross_targets.shift(2).fillna(0.0) * 0.30 * leverage
    fx_position = trend_position + cross_position

    asset_returns = prices.pct_change(fill_method=None)
    phdg_contribution = phdg_position * asset_returns[PHDG]
    trend_contribution = (trend_position * asset_returns.loc[:, FX_SYMBOLS]).sum(axis=1, min_count=1)
    cross_contribution = (cross_position * asset_returns.loc[:, FX_SYMBOLS]).sum(axis=1, min_count=1)
    gross_return = phdg_contribution + trend_contribution + cross_contribution

    phdg_turnover = phdg_position.diff().abs().fillna(0.0)
    fx_turnover = fx_position.diff().abs().sum(axis=1).fillna(0.0)
    trading_cost = phdg_turnover * phdg_cost + fx_turnover * fx_cost
    if phdg_position.iloc[-1] > 0:
        trading_cost.iloc[-1] += phdg_position.iloc[-1] * phdg_cost
    trading_cost.iloc[-1] += fx_position.iloc[-1].abs().sum() * fx_cost
    fx_gross = fx_position.abs().sum(axis=1)
    roll_haircut = fx_gross * annual_roll_haircut / 252.0
    total_gross = phdg_position.abs() + fx_gross
    financing = (total_gross - 1.0).clip(lower=0.0) * annual_financing / 252.0
    raw_net_return = gross_return - trading_cost - roll_haircut - financing
    net_return = apply_bankruptcy(raw_net_return)

    detail = pd.DataFrame({
        "phdg_contribution": phdg_contribution,
        "fx_trend_contribution": trend_contribution,
        "fx_cross_contribution": cross_contribution,
        "gross_return": gross_return, "phdg_turnover": phdg_turnover,
        "fx_turnover": fx_turnover, "trading_cost": trading_cost,
        "fx_gross": fx_gross, "total_gross": total_gross,
        "roll_haircut": roll_haircut, "financing_cost": financing,
        "raw_net_return": raw_net_return, "net_return": net_return,
    }).dropna(subset=["net_return"])
    detail["equity"] = 10000.0 * (1.0 + detail["net_return"]).cumprod()
    detail["bankrupt"] = detail["equity"].eq(0.0)
    positions = pd.concat({
        "fx_trend": trend_position, "fx_cross": cross_position,
        "fx_combined": fx_position,
    }, axis=1)
    positions[("phdg", PHDG)] = phdg_position
    positions = positions.sort_index(axis=1)
    return (
        detail["net_return"], detail, positions.loc[detail.index],
        volatility, trend_momentum, cross_momentum,
    )
