"""Six-coin low-turnover basis, funding-carry and trend portfolio."""
from pathlib import Path

import numpy as np
import pandas as pd

from binance_basis_oos import load_symbol as load_basis_symbol


SYMBOLS = ("VETUSDT", "THETAUSDT", "ALGOUSDT", "ATOMUSDT", "FILUSDT", "UNIUSDT")
LATEST_ACCEPTABLE_START = pd.Timestamp("2020-10-31", tz="UTC")


def load_symbol(symbol, start, end, cache_dir, refresh=False):
    return load_basis_symbol(symbol, start, end, Path(cache_dir), refresh=refresh)


def build_common_panel(frames):
    pieces = []
    for symbol in SYMBOLS:
        if symbol not in frames:
            raise ValueError(f"Missing fixed symbol {symbol}")
        frame = frames[symbol].copy().set_index("funding_time").sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate Binance timestamp for {symbol}")
        if (frame[["spot", "perp_mark"]] <= 0).any().any():
            raise ValueError(f"Nonpositive price for {symbol}")
        pieces.append(frame[["spot", "perp_mark", "funding_rate"]].rename(columns={
            "spot": f"{symbol}__spot", "perp_mark": f"{symbol}__mark",
            "funding_rate": f"{symbol}__funding",
        }))
    return pd.concat(pieces, axis=1, join="inner").dropna().sort_index()


def coverage_audit(panel):
    if panel.empty:
        return {"observations": 0, "start": None, "end": None,
                "coverage_ratio": 0.0, "max_gap_hours": np.inf,
                "scheduled_hours_valid": False, "duplicate_count": 0,
                "span_years": 0.0, "passed": False}
    index = pd.DatetimeIndex(panel.index).tz_convert("UTC")
    expected = pd.date_range(index[0], index[-1], freq="8h", tz="UTC")
    gaps = index.to_series().diff().dropna().dt.total_seconds().div(3600.0)
    audit = {
        "observations": int(len(index)), "start": index[0], "end": index[-1],
        "coverage_ratio": float(len(index.intersection(expected)) / len(expected)),
        "max_gap_hours": float(gaps.max()) if not gaps.empty else 0.0,
        "scheduled_hours_valid": bool(index.hour.isin([0, 8, 16]).all()),
        "duplicate_count": int(index.duplicated().sum()),
        "span_years": float((index[-1] - index[0]).days / 365.25),
    }
    audit["passed"] = bool(
        index[0] <= LATEST_ACCEPTABLE_START and audit["span_years"] >= 5.75
        and audit["coverage_ratio"] >= 0.99 and audit["max_gap_hours"] <= 16.0
        and audit["scheduled_hours_valid"] and audit["duplicate_count"] == 0
    )
    return audit


def _capped_inverse_vol(volatility, signs, cap=0.30):
    vol = pd.Series(volatility, index=SYMBOLS, dtype=float)
    direction = pd.Series(signs, index=SYMBOLS, dtype=float)
    if vol.isna().any() or direction.isna().any() or (vol <= 0).any():
        return pd.Series(np.nan, index=SYMBOLS, dtype=float)
    inverse = 1.0 / vol
    absolute = pd.Series(0.0, index=SYMBOLS)
    remaining, eligible = 1.0, list(SYMBOLS)
    for _ in range(10):
        if not eligible or remaining <= 1e-12:
            break
        proposal = remaining * inverse.loc[eligible] / inverse.loc[eligible].sum()
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
        raise ValueError("Trend cap could not allocate full sleeve")
    return absolute * direction


def sleeve_positions(panel):
    marks = panel[[f"{symbol}__mark" for symbol in SYMBOLS]].copy()
    marks.columns = SYMBOLS
    funding = panel[[f"{symbol}__funding" for symbol in SYMBOLS]].copy()
    funding.columns = SYMBOLS
    update = (panel.index.weekday == 0) & (panel.index.hour == 0)

    basis_spot = pd.DataFrame(0.05, index=panel.index, columns=SYMBOLS)
    basis_perp = pd.DataFrame(-0.05, index=panel.index, columns=SYMBOLS)

    funding_mean = funding.rolling(84, min_periods=84).mean()
    funding_target = pd.DataFrame(np.nan, index=panel.index, columns=SYMBOLS)
    for timestamp in panel.index[update]:
        row = funding_mean.loc[timestamp]
        if row.notna().all():
            ordered = sorted(SYMBOLS, key=lambda symbol: (float(row[symbol]), symbol))
            target = pd.Series(0.0, index=SYMBOLS)
            target.loc[ordered[:2]] = 0.0875
            target.loc[ordered[-2:]] = -0.0875
            funding_target.loc[timestamp] = target
    funding_target = funding_target.ffill().fillna(0.0)

    mark_returns = marks.pct_change(fill_method=None)
    momentum = marks / marks.shift(540) - 1.0
    direction = momentum.gt(0).astype(float) * 2.0 - 1.0
    direction = direction.where(momentum.notna())
    volatility = mark_returns.rolling(180, min_periods=180).std() * np.sqrt(1095.0)
    trend_target = pd.DataFrame(np.nan, index=panel.index, columns=SYMBOLS)
    for timestamp in panel.index[update]:
        if direction.loc[timestamp].notna().all() and volatility.loc[timestamp].notna().all():
            trend_target.loc[timestamp] = 0.35 * _capped_inverse_vol(
                volatility.loc[timestamp], direction.loc[timestamp]
            )
    trend_target = trend_target.ffill().fillna(0.0)
    return {
        "basis_spot": basis_spot.shift(1), "basis_perp": basis_perp.shift(1),
        "funding_perp": funding_target.shift(1), "trend_perp": trend_target.shift(1),
    }, funding_mean, momentum, volatility


def apply_bankruptcy(raw_returns):
    reported = pd.Series(raw_returns).copy()
    bankrupt = reported.le(-1.0)
    if bankrupt.any():
        first = bankrupt[bankrupt].index[0]
        reported.loc[first] = -1.0
        reported.loc[reported.index > first] = 0.0
    return reported


def run_scenario(panel, leverage, leg_cost=0.0005, annual_financing=0.04):
    audit = coverage_audit(panel)
    if not audit["passed"]:
        raise ValueError(f"Low-turnover multicrypto coverage failed: {audit}")
    sleeves, funding_mean, momentum, volatility = sleeve_positions(panel)
    spots = panel[[f"{symbol}__spot" for symbol in SYMBOLS]].copy(); spots.columns = SYMBOLS
    marks = panel[[f"{symbol}__mark" for symbol in SYMBOLS]].copy(); marks.columns = SYMBOLS
    funding = panel[[f"{symbol}__funding" for symbol in SYMBOLS]].copy(); funding.columns = SYMBOLS
    spot_returns = spots.pct_change(fill_method=None)
    mark_returns = marks.pct_change(fill_method=None)
    scaled = {name: weights * leverage for name, weights in sleeves.items()}
    basis = (scaled["basis_spot"] * spot_returns + scaled["basis_perp"] * mark_returns
             - scaled["basis_perp"] * funding).sum(axis=1, min_count=1)
    funding_carry = (scaled["funding_perp"] * mark_returns
                     - scaled["funding_perp"] * funding).sum(axis=1, min_count=1)
    trend = (scaled["trend_perp"] * mark_returns
             - scaled["trend_perp"] * funding).sum(axis=1, min_count=1)
    gross = basis + funding_carry + trend
    spot_position = scaled["basis_spot"]
    perp_position = scaled["basis_perp"] + scaled["funding_perp"] + scaled["trend_perp"]
    spot_turnover = spot_position.diff().abs().sum(axis=1)
    perp_turnover = perp_position.diff().abs().sum(axis=1)
    first = panel.index[1]
    spot_turnover.loc[first] = spot_position.loc[first].abs().sum()
    perp_turnover.loc[first] = perp_position.loc[first].abs().sum()
    trading_cost = (spot_turnover + perp_turnover) * leg_cost
    last = panel.index[-1]
    trading_cost.loc[last] += (spot_position.loc[last].abs().sum()
                               + perp_position.loc[last].abs().sum()) * leg_cost
    financing = max(leverage - 1.0, 0.0) * annual_financing / 1095.0
    net = apply_bankruptcy(gross - trading_cost - financing)
    detail = pd.DataFrame({
        "basis_contribution": basis, "funding_contribution": funding_carry,
        "trend_contribution": trend, "gross_return": gross,
        "spot_turnover": spot_turnover, "perp_turnover": perp_turnover,
        "trading_cost": trading_cost, "financing_cost": financing, "net_return": net,
    }).dropna(subset=["net_return"])
    detail["equity"] = 10000.0 * (1.0 + detail["net_return"]).cumprod()
    detail["bankrupt"] = detail["equity"].eq(0.0)
    daily = detail["net_return"].groupby(detail.index.normalize()).apply(
        lambda values: float((1.0 + values).prod() - 1.0)
    )
    daily.name = "net_return"
    positions = pd.concat({"spot": spot_position, "perp": perp_position}, axis=1)
    return daily, detail, positions.loc[detail.index], sleeves, funding_mean, momentum, volatility
