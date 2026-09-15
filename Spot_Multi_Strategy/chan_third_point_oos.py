"""Causal mechanical proxy for Chan-theory third-class buy/sell points."""
import numpy as np
import pandas as pd

from alphas import chan_theory


SYMBOLS = ("ITOT", "CYB", "DBB", "BIV")
LEG_COST = 0.001
ANNUAL_SHORT_BORROW = 0.01
ANNUAL_FINANCING = 0.04


def adjust_and_validate_ohlc(frame):
    frame = pd.DataFrame(frame).copy().sort_index()
    required = ["raw_open", "raw_high", "raw_low", "raw_close", "adjusted_close"]
    if any(column not in frame for column in required):
        raise ValueError(f"OHLC frame must contain {required}")
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    frame.index = index.normalize()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate calendar date in OHLC frame")
    frame = frame[required]
    values = frame.to_numpy(dtype=float)
    if frame.empty or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("OHLC frame is empty, non-finite or nonpositive")
    adjustment = frame["adjusted_close"] / frame["raw_close"]
    adjusted = pd.DataFrame({
        "open": frame["raw_open"] * adjustment,
        "high": frame["raw_high"] * adjustment,
        "low": frame["raw_low"] * adjustment,
        "close": frame["adjusted_close"],
    }, index=frame.index)
    tolerance = 1e-10
    invalid = (
        (adjusted["low"] > adjusted[["open", "close"]].min(axis=1) + tolerance)
        | (adjusted["high"] + tolerance < adjusted[["open", "close"]].max(axis=1))
        | (adjusted["low"] > adjusted["high"] + tolerance)
    )
    if invalid.any():
        raise ValueError("Adjusted OHLC violates bar ordering")
    return adjusted


def align_markets(raw_by_symbol):
    adjusted = {}
    for symbol in SYMBOLS:
        if symbol not in raw_by_symbol:
            raise ValueError(f"Missing fixed symbol {symbol}")
        adjusted[symbol] = adjust_and_validate_ohlc(raw_by_symbol[symbol])
    common = adjusted[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        common = common.intersection(adjusted[symbol].index)
    common = common.sort_values()
    return {symbol: adjusted[symbol].loc[common].copy() for symbol in SYMBOLS}


def coverage_audit(markets):
    if any(symbol not in markets for symbol in SYMBOLS):
        raise ValueError("Markets do not match frozen four-symbol universe")
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
        index[0] <= pd.Timestamp("2008-06-30", tz="UTC")
        and index[-1] >= pd.Timestamp("2026-08-31", tz="UTC")
        and audit["span_years"] >= 18.0 and audit["coverage_ratio"] >= 0.94
        and audit["max_business_gap"] <= 10 and audit["duplicate_count"] == 0
    )
    return audit


def _update_strokes(points, fractal, min_merged_gap=1):
    merged_idx, kind, price, confirmed_at = fractal
    if not points:
        points.append(fractal)
        return True
    last_idx, last_kind, last_price, _ = points[-1]
    if kind == last_kind:
        more_extreme = ((kind == 1 and price > last_price)
                        or (kind == -1 and price < last_price))
        if more_extreme:
            points[-1] = fractal
            return True
        return False
    if merged_idx - last_idx > min_merged_gap:
        points.append(fractal)
        return True
    return False


def _locked_pivots(points_without_unstable_last):
    pivots = []
    points = points_without_unstable_last
    for k in range(len(points) - 3):
        segment = points[k:k + 4]
        prices = [point[2] for point in segment]
        highs = [max(prices[i], prices[i + 1]) for i in range(3)]
        lows = [min(prices[i], prices[i + 1]) for i in range(3)]
        zg, zd = min(highs), max(lows)
        if zg > zd:
            pivots.append({
                "start_merged_idx": segment[0][0],
                "end_merged_idx": segment[-1][0],
                "zg": float(zg), "zd": float(zd),
            })
    return pivots


def events_from_confirmed_fractals(fractals):
    """Process confirmations in real time; never revise an emitted event."""
    points, events = [], []
    ordered = sorted(fractals, key=lambda item: (item[3], item[0]))
    for fractal in ordered:
        if not _update_strokes(points, fractal) or len(points) < 6:
            continue
        first, retest = points[-2], points[-1]
        eligible = [
            pivot for pivot in _locked_pivots(points[:-1])
            if pivot["end_merged_idx"] < first[0]
        ]
        if not eligible:
            continue
        pivot = eligible[-1]
        signal = 0
        boundary = np.nan
        if first[1] == 1 and retest[1] == -1:
            if first[2] > pivot["zg"] and retest[2] > pivot["zg"]:
                signal, boundary = 1, pivot["zg"]
        elif first[1] == -1 and retest[1] == 1:
            if first[2] < pivot["zd"] and retest[2] < pivot["zd"]:
                signal, boundary = -1, pivot["zd"]
        if signal:
            events.append({
                "confirmed_at": int(retest[3]), "signal": signal,
                "boundary": float(boundary), "zg": pivot["zg"],
                "zd": pivot["zd"], "departure_price": float(first[2]),
                "retest_price": float(retest[2]),
            })
    return pd.DataFrame(events)


def detect_events(frame):
    highs, lows, starts = chan_theory.merge_inclusive_bars(frame)
    fractals = chan_theory.detect_fractals(highs, lows, starts)
    return events_from_confirmed_fractals(fractals)


def build_position(frame):
    frame = pd.DataFrame(frame).copy()
    events = detect_events(frame)
    by_bar = {}
    for row in events.itertuples(index=False):
        by_bar[int(row.confirmed_at)] = row
    target = np.zeros(len(frame), dtype=float)
    boundary = np.full(len(frame), np.nan)
    current, active_boundary = 0.0, np.nan
    close = frame["close"].to_numpy(dtype=float)
    for t in range(len(frame)):
        if current > 0 and close[t] <= active_boundary:
            current, active_boundary = 0.0, np.nan
        elif current < 0 and close[t] >= active_boundary:
            current, active_boundary = 0.0, np.nan
        event = by_bar.get(t)
        if event is not None:
            current = float(event.signal)
            active_boundary = float(event.boundary)
        target[t] = current
        boundary[t] = active_boundary
    target = pd.Series(target, index=frame.index, name="target_position")
    position = target.shift(1).fillna(0.0).rename("position")
    boundary = pd.Series(boundary, index=frame.index, name="active_boundary")
    return position, target, boundary, events


def run_portfolio(markets, leverage, leg_cost=LEG_COST,
                  annual_short_borrow=ANNUAL_SHORT_BORROW,
                  annual_financing=ANNUAL_FINANCING):
    if leverage not in (1.0, 2.0, 3.0, 4.0, 5.0):
        raise ValueError("Leverage must be one of the preregistered scenarios")
    positions, targets, boundaries, events = {}, {}, {}, {}
    asset_returns = {}
    for symbol in SYMBOLS:
        position, target, boundary, event = build_position(markets[symbol])
        positions[symbol], targets[symbol] = position, target
        boundaries[symbol], events[symbol] = boundary, event
        asset_returns[symbol] = markets[symbol]["close"].pct_change(fill_method=None).fillna(0.0)
    positions = pd.DataFrame(positions) * (leverage / len(SYMBOLS))
    targets = pd.DataFrame(targets)
    boundaries = pd.DataFrame(boundaries)
    asset_returns = pd.DataFrame(asset_returns)
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
    slices = [gross_return, turnover, trading_cost, short_borrow,
              financing_cost, gross_exposure, net_return]
    (gross_return, turnover, trading_cost, short_borrow,
     financing_cost, gross_exposure, net_return) = [item.loc[first:] for item in slices]
    positions = positions.loc[first:]
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
    return net_return.rename("net_return"), positions, targets, boundaries, events, detail
