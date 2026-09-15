"""Frozen DAA-G12 replication isolated from the protected core strategy."""
import numpy as np
import pandas as pd


RISKY = (
    "SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO",
    "VNQ", "GSG", "GLD", "TLT", "HYG", "LQD",
)
CANARY = ("VWO", "AGG")
DEFENSIVE = ("SHY", "IEF", "LQD")
SYMBOLS = tuple(dict.fromkeys(RISKY + CANARY + DEFENSIVE))
LEG_COST = 0.001
POST_PUBLICATION_START = pd.Timestamp("2019-01-02", tz="UTC")


def align_prices(prices_by_symbol):
    normalized = {}
    for symbol in SYMBOLS:
        if symbol not in prices_by_symbol:
            raise ValueError(f"Missing fixed symbol {symbol}")
        series = pd.Series(prices_by_symbol[symbol]).dropna().astype(float).sort_index()
        index = pd.DatetimeIndex(series.index)
        index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        index = index.normalize()
        if index.has_duplicates:
            raise ValueError(f"Duplicate calendar-date price for {symbol}")
        if (series <= 0).any():
            raise ValueError(f"Nonpositive price for {symbol}")
        normalized[symbol] = pd.Series(series.to_numpy(), index=index, name=symbol)
    return pd.concat(normalized.values(), axis=1, join="inner").dropna().sort_index()


def coverage_audit(prices):
    if prices.empty:
        return {"observations": 0, "start": None, "end": None,
                "coverage_ratio": 0.0, "max_business_gap": np.inf,
                "duplicate_count": 0, "span_years": 0.0, "passed": False}
    if list(prices.columns) != list(SYMBOLS):
        raise ValueError("Panel columns do not match frozen DAA-G12 universe")
    index = pd.DatetimeIndex(prices.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    expected = pd.bdate_range(index[0], index[-1], tz="UTC")
    locations = expected.get_indexer(index)
    max_gap = int(np.diff(locations).max() - 1) if len(locations) > 1 else 0
    audit = {
        "observations": int(len(index)), "start": index[0], "end": index[-1],
        "coverage_ratio": float(len(index.intersection(expected)) / len(expected)),
        "max_business_gap": max_gap, "duplicate_count": int(index.duplicated().sum()),
        "span_years": float((index[-1] - index[0]).days / 365.25),
    }
    audit["passed"] = bool(
        index[0] <= pd.Timestamp("2006-08-31", tz="UTC")
        and index[-1] >= pd.Timestamp("2026-08-31", tz="UTC")
        and audit["span_years"] >= 20.0
        and audit["coverage_ratio"] >= 0.94 and audit["max_business_gap"] <= 10
        and audit["duplicate_count"] == 0 and not prices.isna().any().any()
        and (prices > 0).all().all()
    )
    return audit


def monthly_levels(prices):
    keys = pd.DatetimeIndex(prices.index).tz_localize(None).to_period("M")
    return prices.groupby(keys, sort=True).tail(1)


def momentum_13612w(monthly):
    monthly = pd.DataFrame(monthly).astype(float)
    return sum(
        weight * (monthly / monthly.shift(lag) - 1.0)
        for lag, weight in ((1, 12.0), (3, 4.0), (6, 2.0), (12, 1.0))
    )


def _rank(row, universe):
    order = {symbol: position for position, symbol in enumerate(universe)}
    return sorted(universe, key=lambda symbol: (-float(row[symbol]), order[symbol]))


def build_month_end_weights(prices):
    scores = momentum_13612w(monthly_levels(prices)).dropna(how="any")
    weights = pd.DataFrame(0.0, index=scores.index, columns=SYMBOLS)
    for date, score in scores.iterrows():
        bad = sum(float(score[symbol]) <= 0.0 for symbol in CANARY)
        defensive_fraction = min(bad / 2.0, 1.0)
        risky_count = int(round(6 * (1.0 - defensive_fraction)))
        for symbol in _rank(score, RISKY)[:risky_count]:
            weights.at[date, symbol] += 1.0 / 6.0
        if defensive_fraction > 0.0:
            best_defensive = _rank(score, DEFENSIVE)[0]
            weights.at[date, best_defensive] += defensive_fraction
    return weights, scores


def build_daily_positions(prices, leverage=1.0):
    if leverage not in (1.0, 2.0, 3.0):
        raise ValueError("Only preregistered leverage 1x, 2x or 3x is allowed")
    month_end, scores = build_month_end_weights(prices)
    positions = month_end.reindex(prices.index).ffill().shift(1).fillna(0.0)
    return positions * leverage, month_end * leverage, scores


def run_backtest(prices, leverage=1.0, leg_cost=LEG_COST,
                 annual_financing=0.04):
    prices = pd.DataFrame(prices).astype(float).sort_index()
    positions, month_end_weights, scores = build_daily_positions(prices, leverage)
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    gross_return = (positions * asset_returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    trading_cost = turnover * leg_cost
    gross_exposure = positions.abs().sum(axis=1)
    financing_cost = (gross_exposure - 1.0).clip(lower=0.0) * annual_financing / 252.0
    net_return = gross_return - trading_cost - financing_cost

    active = gross_exposure.gt(0.0)
    if not active.any():
        raise ValueError("No DAA-G12 allocation after twelve-month warm-up")
    first = active[active].index[0]
    net_return = net_return.loc[first:]
    gross_return = gross_return.loc[first:]
    turnover = turnover.loc[first:]
    trading_cost = trading_cost.loc[first:]
    financing_cost = financing_cost.loc[first:]
    positions = positions.loc[first:]
    gross_exposure = gross_exposure.loc[first:]
    trading_cost.iloc[-1] += positions.iloc[-1].abs().sum() * leg_cost
    net_return.iloc[-1] -= positions.iloc[-1].abs().sum() * leg_cost

    bankrupt = net_return.le(-1.0)
    if bankrupt.any():
        failure = bankrupt[bankrupt].index[0]
        net_return.loc[failure] = -1.0
        net_return.loc[net_return.index > failure] = 0.0
    detail = pd.DataFrame({
        "gross_return": gross_return, "turnover": turnover,
        "trading_cost": trading_cost, "financing_cost": financing_cost,
        "gross_exposure": gross_exposure, "net_return": net_return,
    })
    return net_return.rename("net_return"), positions, month_end_weights, scores, detail
