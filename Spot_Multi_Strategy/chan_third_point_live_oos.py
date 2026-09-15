"""Frozen Chan third-point engine applied to four currently live ETFs."""
import numpy as np
import pandas as pd

import chan_third_point_oos as core


SYMBOLS = ("SCHB", "USDU", "DBB", "BIV")


def align_markets(raw_by_symbol):
    adjusted = {}
    for symbol in SYMBOLS:
        if symbol not in raw_by_symbol:
            raise ValueError(f"Missing fixed symbol {symbol}")
        adjusted[symbol] = core.adjust_and_validate_ohlc(raw_by_symbol[symbol])
    common = adjusted[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        common = common.intersection(adjusted[symbol].index)
    common = common.sort_values()
    return {symbol: adjusted[symbol].loc[common].copy() for symbol in SYMBOLS}


def coverage_audit(markets):
    if any(symbol not in markets for symbol in SYMBOLS):
        raise ValueError("Markets do not match frozen live-ETF universe")
    index = markets[SYMBOLS[0]].index
    if any(not index.equals(markets[symbol].index) for symbol in SYMBOLS[1:]):
        raise ValueError("Markets are not on one common calendar")
    if len(index) == 0:
        return {"observations": 0, "start": None, "end": None,
                "coverage_ratio": 0.0, "max_business_gap": np.inf,
                "duplicate_count": 0, "span_years": 0.0, "passed": False}
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
        index[0] <= pd.Timestamp("2014-01-31", tz="UTC")
        and index[-1] >= pd.Timestamp("2026-08-31", tz="UTC")
        and audit["span_years"] >= 12.5 and audit["coverage_ratio"] >= 0.94
        and audit["max_business_gap"] <= 10 and audit["duplicate_count"] == 0
    )
    return audit


def run_portfolio(markets, leverage, leg_cost=core.LEG_COST,
                  annual_short_borrow=core.ANNUAL_SHORT_BORROW,
                  annual_financing=core.ANNUAL_FINANCING):
    if leverage not in (1.0, 2.0, 3.0, 4.0, 5.0):
        raise ValueError("Leverage must be one of the preregistered scenarios")
    positions, targets, boundaries, events, asset_returns = {}, {}, {}, {}, {}
    for symbol in SYMBOLS:
        position, target, boundary, event = core.build_position(markets[symbol])
        positions[symbol], targets[symbol] = position, target
        boundaries[symbol], events[symbol] = boundary, event
        asset_returns[symbol] = markets[symbol]["close"].pct_change(fill_method=None).fillna(0.0)
    unit_positions = pd.DataFrame(positions)
    positions = unit_positions * (leverage / len(SYMBOLS))
    targets = pd.DataFrame(targets)
    boundaries = pd.DataFrame(boundaries)
    asset_returns = pd.DataFrame(asset_returns)
    positioned_returns = unit_positions * asset_returns
    gross_return = (positions * asset_returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    trading_cost = turnover * leg_cost
    short_exposure = positions.clip(upper=0.0).abs().sum(axis=1)
    gross_exposure = positions.abs().sum(axis=1)
    short_borrow = short_exposure * annual_short_borrow / 252.0
    financing_cost = (gross_exposure - 1.0).clip(lower=0.0) * annual_financing / 252.0
    net_return = gross_return - trading_cost - short_borrow - financing_cost
    active = gross_exposure.gt(0.0)
    if not active.any():
        raise ValueError("No third-point signal in the fixed sample")
    first = active[active].index[0]
    series = [gross_return, turnover, trading_cost, short_borrow,
              financing_cost, gross_exposure, net_return]
    (gross_return, turnover, trading_cost, short_borrow,
     financing_cost, gross_exposure, net_return) = [item.loc[first:] for item in series]
    positions = positions.loc[first:]
    positioned_returns = positioned_returns.loc[first:]
    final_liquidation = positions.iloc[-1].abs().sum() * leg_cost
    trading_cost.iloc[-1] += final_liquidation
    net_return.iloc[-1] -= final_liquidation
    bankrupt = net_return.le(-1.0)
    if bankrupt.any():
        failure = bankrupt[bankrupt].index[0]
        net_return.loc[failure] = -1.0
        net_return.loc[net_return.index > failure] = 0.0
    detail = pd.DataFrame({
        "gross_return": gross_return, "turnover": turnover,
        "trading_cost": trading_cost, "short_borrow": short_borrow,
        "financing_cost": financing_cost, "gross_exposure": gross_exposure,
        "net_return": net_return,
    })
    return (net_return.rename("net_return"), positions, targets, boundaries,
            events, positioned_returns, detail)


def one_x_sleeve_attribution(markets, positions, leg_cost=core.LEG_COST,
                             annual_short_borrow=core.ANNUAL_SHORT_BORROW):
    """Allocate the exact 1x portfolio PnL and costs back to each sleeve."""
    if not np.allclose(positions.abs().sum(axis=1).to_numpy(),
                       positions.abs().sum(axis=1).clip(upper=1.0).to_numpy()):
        raise ValueError("Sleeve attribution is frozen to the 1x scenario")
    rows = {}
    for symbol in SYMBOLS:
        asset_return = markets[symbol]["close"].pct_change(fill_method=None).fillna(0.0)
        weight = positions[symbol]
        gross = weight * asset_return.reindex(weight.index)
        turnover = weight.diff().abs()
        turnover.iloc[0] = abs(weight.iloc[0])
        trading_cost = turnover * leg_cost
        trading_cost.iloc[-1] += abs(weight.iloc[-1]) * leg_cost
        short_borrow = weight.clip(upper=0.0).abs() * annual_short_borrow / 252.0
        rows[symbol] = gross - trading_cost - short_borrow
    return pd.DataFrame(rows)
