"""Frozen execution ledger for the preregistered Fed-reserves crypto study.

Inputs are already-audited, in-memory vintage observations and official market
frames. This module fetches nothing and cannot select assets or parameters.
The return for day D spans D 00:00 to D+1 00:00 UTC; settlement and order
costs on the right boundary belong to that daily return. Intrabar favorable
peaks are deliberately ordered before adverse troughs for conservative risk.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date
import math
from statistics import median
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

if __package__:
    from .wiki_attention_sim import (
        DailyRecord, Ledger, MarginRuinError, SimulationDataError, Trade,
    )
else:
    from wiki_attention_sim import (
        DailyRecord, Ledger, MarginRuinError, SimulationDataError, Trade,
    )


SYMBOLS = ("BCHUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LTCUSDT")
LOTS = {"BCHUSDT": 0.01, "BTCUSDT": 0.001, "DOGEUSDT": 1.0,
        "ETHUSDT": 0.001, "LTCUSDT": 0.01}
WARMUP = pd.Timestamp("2020-12-01 00:00:00", tz="UTC")
START = pd.Timestamp("2021-01-01 00:00:00", tz="UTC")
END = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
INITIAL_EQUITY = 10_000.0
FEE_RATE = 0.001
MIN_ORDER_USDT = 5.0
MAX_CAPACITY_RATIO = 0.01
MAINTENANCE_PROXY = 0.05
VOL_TARGET = 0.30
GROSS_CAP = 3.0
ANNUAL_DAYS = 365.25


def _utc_timestamp(value: object, label: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SimulationDataError(f"{label}: invalid timestamp") from exc
    if stamp.tzinfo is None or str(stamp.tz) != "UTC":
        raise SimulationDataError(f"{label}: UTC-aware timestamp required")
    return stamp


def _observation_field(observation: object, name: str) -> object:
    if isinstance(observation, Mapping):
        if name not in observation:
            raise SimulationDataError(f"H.4.1 observation lacks {name}")
        return observation[name]
    try:
        return getattr(observation, name)
    except AttributeError as exc:
        raise SimulationDataError(f"H.4.1 observation lacks {name}") from exc


def _validate_observations(observations: Sequence[object]) -> tuple[tuple[pd.Timestamp, ...], tuple[float, ...]]:
    if not isinstance(observations, Sequence) or len(observations) < 5:
        raise SimulationDataError("H.4.1 series needs at least five releases")
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    previous_observed: pd.Timestamp | None = None
    for item in observations:
        published = _utc_timestamp(_observation_field(item, "release_date"), "release_date")
        observed = _utc_timestamp(_observation_field(item, "observed_date"), "observed_date")
        if (published != published.normalize() or observed != observed.normalize() or
                observed.weekday() != 2 or observed >= published):
            raise SimulationDataError("H.4.1 release/Wednesday observation date invalid")
        if dates and published <= dates[-1]:
            raise SimulationDataError("H.4.1 release dates unordered or duplicate")
        if previous_observed is not None and observed <= previous_observed:
            raise SimulationDataError("H.4.1 observation dates unordered or duplicate")
        try:
            value = float(_observation_field(item, "value"))
        except (TypeError, ValueError) as exc:
            raise SimulationDataError("H.4.1 balance not numeric") from exc
        if not math.isfinite(value) or value < 0:
            raise SimulationDataError("H.4.1 balance nonfinite or negative")
        dates.append(published)
        values.append(value)
        previous_observed = observed
    return tuple(dates), tuple(values)


def _require_symbols(frames: Mapping[str, object], label: str) -> None:
    if not isinstance(frames, Mapping) or set(frames) != set(SYMBOLS):
        raise SimulationDataError(f"{label} must contain exactly {SYMBOLS}")


def _validate_bars(frames: Mapping[str, pd.DataFrame], *, trade: bool,
                   grid: pd.DatetimeIndex) -> dict[str, dict[str, np.ndarray]]:
    label = "trade_bars" if trade else "mark_bars"
    _require_symbols(frames, label)
    required = ("open", "high", "low", "close", "quote_volume") if trade else (
        "open", "high", "low", "close")
    checked: dict[str, dict[str, np.ndarray]] = {}
    for symbol in SYMBOLS:
        frame = frames[symbol]
        if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
            raise SimulationDataError(f"{symbol}: {label} must be a dated DataFrame")
        if not frame.index.equals(grid):
            raise SimulationDataError(f"{symbol}: missing, duplicate or non-UTC 4h {label} bar")
        if not frame.columns.is_unique or not set(required).issubset(frame.columns):
            raise SimulationDataError(f"{symbol}: {label} lacks unique required OHLC/volume columns")
        try:
            cols = {key: frame[key].to_numpy(dtype=float, na_value=np.nan)
                    for key in required}
        except (TypeError, ValueError) as exc:
            raise SimulationDataError(f"{symbol}: nonnumeric {label} field") from exc
        o, h, l, c = (cols[key] for key in ("open", "high", "low", "close"))
        if (not np.isfinite(o).all() or np.any(o <= 0) or
                not all(np.isfinite(arr[:-1]).all() for arr in (h, l, c)) or
                np.any(np.minimum.reduce((o[:-1], h[:-1], l[:-1], c[:-1])) <= 0) or
                np.any(h[:-1] < np.maximum.reduce((o[:-1], l[:-1], c[:-1]))) or
                np.any(l[:-1] > np.minimum.reduce((o[:-1], h[:-1], c[:-1])))):
            raise SimulationDataError(f"{symbol}: invalid {label} OHLC")
        if not all(np.isnan(cols[key][-1]) for key in ("high", "low", "close")):
            raise SimulationDataError(f"{symbol}: terminal {label} bar reveals future OHLC")
        if trade:
            volume = cols["quote_volume"]
            if (not np.isfinite(volume[:-1]).all() or np.any(volume[:-1] < 0) or
                    not np.isnan(volume[-1])):
                raise SimulationDataError(f"{symbol}: invalid {label} quote volume")
        checked[symbol] = cols
    return checked


def _validate_funding(frames: Mapping[str, pd.DataFrame],
                      grid: pd.DatetimeIndex) -> dict[str, dict[int, float]]:
    _require_symbols(frames, "funding_events")
    output: dict[str, dict[int, float]] = {}
    for symbol in SYMBOLS:
        frame = frames[symbol]
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not {
                "funding_time", "rate", "funding_interval_hours"}.issubset(frame.columns):
            raise SimulationDataError(f"{symbol}: malformed funding frame")
        times = frame["funding_time"]
        if not isinstance(times.dtype, pd.DatetimeTZDtype):
            raise SimulationDataError(f"{symbol}: funding times must be UTC-aware")
        stamps = pd.DatetimeIndex(times)
        if (len(stamps) < 2 or stamps.has_duplicates or not stamps.is_monotonic_increasing or
                str(stamps.tz) != "UTC"):
            raise SimulationDataError(f"{symbol}: funding events unordered, duplicate or non-UTC")
        if "raw_funding_time" in frame:
            raw = frame["raw_funding_time"]
            if not isinstance(raw.dtype, pd.DatetimeTZDtype):
                raise SimulationDataError(f"{symbol}: raw funding time must be UTC-aware")
            raw_stamps = pd.DatetimeIndex(raw)
            offsets = np.asarray((raw_stamps - stamps) /
                                 pd.Timedelta(milliseconds=1), dtype=float)
            if (str(raw_stamps.tz) != "UTC" or not np.isfinite(offsets).all() or
                    np.any(np.abs(offsets) > 100)):
                raise SimulationDataError(f"{symbol}: funding normalization exceeds 100ms")
        if stamps[0] > START or stamps[-1] != END:
            raise SimulationDataError(f"{symbol}: funding does not cover start and closeout")
        positions = grid.get_indexer(stamps)
        if np.any(positions < 0):
            raise SimulationDataError(f"{symbol}: funding not on exact 4h mark open")
        try:
            rates = frame["rate"].to_numpy(dtype=float, na_value=np.nan)
            intervals = frame["funding_interval_hours"].to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise SimulationDataError(f"{symbol}: nonnumeric funding rate/interval") from exc
        gaps = np.asarray((stamps[1:] - stamps[:-1]) / pd.Timedelta(hours=1), dtype=float)
        if (not np.isfinite(rates).all() or not np.isfinite(intervals).all() or
                np.any(intervals <= 0) or np.any(intervals > 24) or
                not np.allclose(gaps, intervals[1:], rtol=0, atol=1e-9)):
            raise SimulationDataError(f"{symbol}: incomplete funding event sequence")
        output[symbol] = dict(zip(positions.tolist(), rates.tolist(), strict=True))
    return output


def _direction(releases: Sequence[pd.Timestamp], values: Sequence[float],
               monday: pd.Timestamp) -> int:
    """Four releases, not four calendar weeks, with a four-day publication lag."""
    if monday.weekday() != 0 or monday.hour != 0:
        raise ValueError("decisions are Monday 00:00 UTC only")
    latest = bisect_right(releases, monday - pd.Timedelta(days=4)) - 1
    if latest < 4:
        raise SimulationDataError(f"insufficient H.4.1 release history at {monday}")
    return (values[latest] > values[latest - 4]) - (
        values[latest] < values[latest - 4])


def _realized_vol(trade: Mapping[str, Mapping[str, np.ndarray]],
                  day_number: int) -> float:
    """Sample std of prior 30 completed UTC day equal-weight basket returns."""
    if day_number < 31:
        raise SimulationDataError("less than 30 complete daily returns for sizing")
    basket = np.zeros(30, dtype=float)
    for symbol in SYMBOLS:
        closes = trade[symbol]["close"]
        current = closes[(day_number - 30) * 6 + 5:day_number * 6:6]
        previous = closes[(day_number - 31) * 6 + 5:(day_number - 1) * 6:6]
        if len(current) != 30 or len(previous) != 30:
            raise SimulationDataError("incomplete 30-day sizing window")
        basket += (current / previous - 1.0) / len(SYMBOLS)
    volatility = float(np.std(basket, ddof=1) * math.sqrt(ANNUAL_DAYS))
    if not math.isfinite(volatility) or volatility <= 0:
        raise SimulationDataError("nonfinite or zero prior-30-day basket volatility")
    return volatility


def run_backtest(observations: Sequence[object],
                 trade_bars: Mapping[str, pd.DataFrame],
                 mark_bars: Mapping[str, pd.DataFrame],
                 funding_events: Mapping[str, pd.DataFrame], *,
                 slippage_rate: float = 0.001, benchmark: bool = False) -> Ledger:
    """Execute one frozen Monday H.4.1 rule or its same-risk always-long control."""
    if slippage_rate not in (0.001, 0.0025) or type(benchmark) is not bool:
        raise ValueError("only registered 10/25bp slippage and boolean control allowed")
    releases, values = _validate_observations(observations)
    grid = pd.date_range(WARMUP, END, freq="4h", tz="UTC")
    trade = _validate_bars(trade_bars, trade=True, grid=grid)
    mark = _validate_bars(mark_bars, trade=False, grid=grid)
    funding = _validate_funding(funding_events, grid)
    daily_volumes = {s: trade[s]["quote_volume"][:-1].reshape(-1, 6).sum(axis=1)
                     for s in SYMBOLS}
    start_index = int(grid.get_loc(START))
    end_index = len(grid) - 1

    cash = INITIAL_EQUITY
    holdings = {s: 0 for s in SYMBOLS}  # signed integer lot counts
    trades: list[Trade] = []
    fee_total = slip_total = funding_cost = turnover = 0.0
    capacity_breaches = execution_breaches = 0
    daily_open: dict[date, float] = {}
    daily_high: dict[date, float] = {}
    daily_low: dict[date, float] = {}
    daily_exposed: dict[date, bool] = {}
    all_time_peak = INITIAL_EQUITY
    worst_drawdown = 0.0
    annual_year = START.year
    annual_peak = INITIAL_EQUITY
    annual_drawdown: dict[int, float] = {annual_year: 0.0}

    def current_equity(index: int) -> float:
        return cash + sum(holdings[s] * LOTS[s] * mark[s]["open"][index]
                          for s in SYMBOLS)

    def gross_notional(index: int) -> float:
        return sum(abs(holdings[s] * LOTS[s] * mark[s]["open"][index])
                   for s in SYMBOLS)

    def observed_risk(index: int, stamp: pd.Timestamp) -> float:
        value = current_equity(index)
        gross = gross_notional(index)
        if not math.isfinite(value) or value <= 0:
            raise MarginRuinError(stamp, value, value, "nonpositive observed equity")
        if gross > 0 and value < MAINTENANCE_PROXY * gross:
            raise MarginRuinError(stamp, value, value, "observed maintenance proxy breach")
        return value

    def trade_lots(symbol: str, delta_lots: int, index: int,
                   stamp: pd.Timestamp) -> None:
        nonlocal cash, fee_total, slip_total, turnover
        nonlocal capacity_breaches, execution_breaches
        if delta_lots == 0:
            return
        quantity = delta_lots * LOTS[symbol]
        reference = float(trade[symbol]["open"][index])
        fill = reference * (1.0 + slippage_rate if delta_lots > 0 else
                            1.0 - slippage_rate)
        notional = abs(quantity * fill)
        if notional < MIN_ORDER_USDT:
            execution_breaches += 1
            return
        past_day = index // 6
        if past_day < 30:
            raise SimulationDataError("less than 30 complete days of liquidity warmup")
        typical = median(daily_volumes[symbol][past_day - 30:past_day].tolist())
        ratio = notional / typical if typical > 0 else math.inf
        if ratio > MAX_CAPACITY_RATIO:
            capacity_breaches += 1
        fee = FEE_RATE * notional
        cash -= quantity * fill + fee
        holdings[symbol] += delta_lots
        fee_total += fee
        slip_total += abs(quantity) * abs(fill - reference)
        turnover += notional
        trades.append(Trade(stamp.to_pydatetime(), symbol,
                            "buy" if delta_lots > 0 else "sell", quantity,
                            reference, fill, notional, fee, ratio))

    def rebalance(target: Mapping[str, float], index: int,
                  stamp: pd.Timestamp) -> None:
        nonlocal execution_breaches
        size_equity = observed_risk(index, stamp)
        target_lots: dict[str, int] = {}
        for symbol in SYMBOLS:
            desired = target[symbol]
            price = float(trade[symbol]["open"][index])
            whole = math.floor(abs(desired) * size_equity /
                               (price * LOTS[symbol]) + 1e-12)
            if desired != 0 and whole == 0:
                execution_breaches += 1
            target_lots[symbol] = (1 if desired > 0 else -1) * whole if desired else 0

        # Reduce or close first, then add. A sub-minimum residual cannot cross.
        for symbol in SYMBOLS:
            old, new = holdings[symbol], target_lots[symbol]
            if old and (new == 0 or old * new < 0):
                trade_lots(symbol, -old, index, stamp)
            elif old and new * old > 0 and abs(new) < abs(old):
                trade_lots(symbol, new - old, index, stamp)
        for symbol in SYMBOLS:
            old, new = holdings[symbol], target_lots[symbol]
            if new and old * new < 0:
                execution_breaches += 1
                continue
            if new and (old == 0 or new * old > 0 and abs(new) > abs(old)):
                trade_lots(symbol, new - old, index, stamp)

    def ordered_drawdown(value: float, *, high: bool, year: int) -> None:
        nonlocal all_time_peak, worst_drawdown, annual_peak
        if high:
            all_time_peak = max(all_time_peak, value)
            annual_peak = max(annual_peak, value)
        else:
            worst_drawdown = min(worst_drawdown, value / all_time_peak - 1.0)
            annual_drawdown[year] = min(annual_drawdown[year],
                                        value / annual_peak - 1.0)

    for index in range(start_index, end_index + 1):
        stamp = grid[index]
        day = stamp.date()
        before_funding = observed_risk(index, stamp)
        ordered_drawdown(before_funding, high=True, year=annual_year)
        ordered_drawdown(before_funding, high=False, year=annual_year)

        # The old holdings pay/receive funding before Monday orders and closeout.
        for symbol in SYMBOLS:
            rate = funding[symbol].get(index)
            if rate is not None and holdings[symbol]:
                payment = holdings[symbol] * LOTS[symbol] * mark[symbol]["open"][index] * rate
                cash -= payment
                funding_cost += payment
        observed_risk(index, stamp)

        if stamp == END:
            rebalance({s: 0.0 for s in SYMBOLS}, index, stamp)
        elif stamp.weekday() == 0 and stamp.hour == 0:
            direction = 1 if benchmark else _direction(releases, values, stamp)
            gross = min(GROSS_CAP, VOL_TARGET / _realized_vol(trade, index // 6))
            rebalance({s: direction * gross / len(SYMBOLS) for s in SYMBOLS},
                      index, stamp)

        after_trade = observed_risk(index, stamp)
        ordered_drawdown(after_trade, high=True, year=annual_year)
        ordered_drawdown(after_trade, high=False, year=annual_year)
        if stamp.hour == 0:
            daily_open[day] = INITIAL_EQUITY if stamp == START else after_trade
            daily_exposed[day] = any(holdings.values())
        if stamp == END:
            break

        if stamp.year != annual_year:
            annual_year = stamp.year
            annual_peak = after_trade
            annual_drawdown[annual_year] = 0.0

        high_equity = cash
        low_equity = cash
        stressed_gross = 0.0
        for symbol in SYMBOLS:
            quantity = holdings[symbol] * LOTS[symbol]
            if quantity >= 0:
                high_equity += quantity * mark[symbol]["high"][index]
                low_equity += quantity * mark[symbol]["low"][index]
                stressed_gross += abs(quantity * mark[symbol]["low"][index])
            else:
                high_equity += quantity * mark[symbol]["low"][index]
                low_equity += quantity * mark[symbol]["high"][index]
                stressed_gross += abs(quantity * mark[symbol]["high"][index])
        if not all(math.isfinite(v) for v in (high_equity, low_equity)):
            raise SimulationDataError("nonfinite stressed mark")
        if (low_equity <= 0 or stressed_gross > 0 and
                low_equity < MAINTENANCE_PROXY * stressed_gross):
            raise MarginRuinError(stamp, after_trade, low_equity,
                                  "adverse 4h maintenance proxy breach")
        ordered_drawdown(high_equity, high=True, year=annual_year)
        ordered_drawdown(low_equity, high=False, year=annual_year)
        daily_high[day] = max(daily_high.get(day, -math.inf), before_funding,
                              after_trade, high_equity)
        daily_low[day] = min(daily_low.get(day, math.inf), before_funding,
                             after_trade, low_equity)

    records: list[DailyRecord] = []
    day = START
    while day < END:
        next_day = day + pd.Timedelta(days=1)
        opened = daily_open[day.date()]
        closed = daily_open[next_day.date()]
        records.append(DailyRecord(day.date(), opened, closed,
                                   closed / opened - 1.0,
                                   daily_high[day.date()], daily_low[day.date()],
                                   daily_exposed[day.date()]))
        day = next_day
    return Ledger(tuple(records), current_equity(end_index), fee_total,
                  slip_total, funding_cost, turnover, tuple(trades),
                  capacity_breaches, execution_breaches, 0, worst_drawdown,
                  annual_drawdown, not any(holdings.values()))
