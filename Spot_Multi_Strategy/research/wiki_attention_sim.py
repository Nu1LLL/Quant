"""Frozen execution ledger for the preregistered five-coin Wikipedia study.

This module accepts already-audited in-memory source frames.  It never fetches
data, selects assets, or changes the rule in response to a result.  A return
labelled D spans [D 00:00 UTC, D+1 00:00 UTC); costs at the right-hand open
therefore belong to that return.  Funding is a signed *cost*: positive means
paid by the account, negative means received. Fees and turnover use executed
fill notional; slippage is the nonnegative fill-versus-reference loss. Daily
stress extremes also include the pre-settlement 00:00 equity, conservatively.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from statistics import median
from typing import Mapping

import numpy as np
import pandas as pd


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


class SimulationDataError(ValueError):
    """An input violates the registered source or grid contract."""


class MarginRuinError(RuntimeError):
    """Observed or adverse-bar equity cannot support continued hypothetical trading."""

    def __init__(self, timestamp: pd.Timestamp, equity: float,
                 stressed_equity: float, reason: str) -> None:
        self.timestamp = timestamp
        self.day = timestamp.date()
        self.equity = equity
        self.stressed_equity = stressed_equity
        self.reason = reason
        super().__init__(f"{reason} at {timestamp.isoformat()}: "
                         f"equity={equity:.8f}, stressed={stressed_equity:.8f}")


@dataclass(frozen=True)
class Trade:
    time: datetime
    symbol: str
    side: str
    quantity: float
    reference_price: float
    fill_price: float
    fill_notional_usdt: float
    fee_usdt: float
    capacity_ratio: float


@dataclass(frozen=True)
class DailyRecord:
    day: date
    equity_open: float
    equity_next_open: float
    return_net: float
    conservative_high_equity: float
    conservative_low_equity: float
    exposed: bool


@dataclass(frozen=True)
class Ledger:
    records: tuple[DailyRecord, ...]
    final_equity: float
    fee_usdt: float
    slippage_usdt: float
    funding_usdt: float
    turnover_usdt: float
    trades: tuple[Trade, ...]
    capacity_breaches: int
    execution_breaches: int
    margin_proxy_breaches: int
    conservative_max_drawdown: float
    conservative_max_drawdown_by_year: dict[int, float]
    terminal_flat: bool


def _require_keys(actual: Mapping[str, object], name: str) -> None:
    if not isinstance(actual, Mapping) or set(actual) != set(SYMBOLS):
        raise SimulationDataError(f"{name} must contain exactly {SYMBOLS}")


def _validate_views(views: Mapping[str, pd.Series]) -> dict[str, pd.Series]:
    _require_keys(views, "views")
    expected = pd.date_range(WARMUP, END - pd.Timedelta(days=1),
                             freq="D", tz="UTC")
    output: dict[str, pd.Series] = {}
    for symbol in SYMBOLS:
        series = views[symbol]
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
            raise SimulationDataError(f"{symbol}: views must be a dated Series")
        if not series.index.equals(expected):
            raise SimulationDataError(f"{symbol}: missing, duplicate or non-UTC view day")
        try:
            values = series.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise SimulationDataError(f"{symbol}: nonnumeric pageviews") from exc
        if (not np.isfinite(values).all() or np.any(values <= 0) or
                not np.equal(values, np.floor(values)).all()):
            raise SimulationDataError(f"{symbol}: pageviews must be positive integers")
        output[symbol] = pd.Series(values, index=expected)
    return output


def _validate_bars(frames: Mapping[str, pd.DataFrame], *,
                   trade: bool) -> dict[str, dict[str, np.ndarray]]:
    name = "trade_bars" if trade else "mark_bars"
    _require_keys(frames, name)
    expected = pd.date_range(WARMUP, END, freq="4h", tz="UTC")
    required = ("open", "high", "low", "close", "quote_volume") if trade else (
        "open", "high", "low", "close")
    output: dict[str, dict[str, np.ndarray]] = {}
    for symbol in SYMBOLS:
        frame = frames[symbol]
        if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
            raise SimulationDataError(f"{symbol}: {name} must be a dated DataFrame")
        if not frame.columns.is_unique:
            raise SimulationDataError(f"{symbol}: duplicate {name} column")
        if not frame.index.equals(expected):
            raise SimulationDataError(f"{symbol}: missing, duplicate or non-UTC 4h {name} bar")
        if not set(required).issubset(frame.columns):
            raise SimulationDataError(f"{symbol}: {name} lacks required OHLC/volume columns")
        try:
            columns = {key: frame[key].to_numpy(dtype=float, na_value=np.nan)
                       for key in required}
        except (TypeError, ValueError) as exc:
            raise SimulationDataError(f"{symbol}: nonnumeric {name} field") from exc
        o, h, l, c = (columns[key] for key in ("open", "high", "low", "close"))
        if (not np.isfinite(o).all() or np.any(o <= 0) or
                not all(np.isfinite(arr[:-1]).all() for arr in (h, l, c)) or
                np.any(np.minimum.reduce((o[:-1], h[:-1], l[:-1], c[:-1])) <= 0) or
                np.any(h[:-1] < np.maximum.reduce((o[:-1], l[:-1], c[:-1]))) or
                np.any(l[:-1] > np.minimum.reduce((o[:-1], h[:-1], c[:-1])))):
            raise SimulationDataError(f"{symbol}: invalid {name} OHLC")
        # At the 2026-01-01 closeout we may inspect only that bar's open.
        if not all(np.isnan(columns[key][-1]) for key in ("high", "low", "close")):
            raise SimulationDataError(f"{symbol}: terminal {name} bar reveals future OHLC")
        if trade:
            volume = columns["quote_volume"]
            if (not np.isfinite(volume[:-1]).all() or np.any(volume[:-1] < 0) or
                    not np.isnan(volume[-1])):
                raise SimulationDataError(f"{symbol}: invalid {name} quote volume")
        output[symbol] = columns
    return output


def _validate_funding(frames: Mapping[str, pd.DataFrame],
                      grid: pd.DatetimeIndex) -> dict[str, dict[int, float]]:
    _require_keys(frames, "funding_events")
    output: dict[str, dict[int, float]] = {}
    for symbol in SYMBOLS:
        frame = frames[symbol]
        if not isinstance(frame, pd.DataFrame) or not {
                "funding_time", "rate", "funding_interval_hours"}.issubset(frame.columns):
            raise SimulationDataError(f"{symbol}: malformed funding frame")
        if not frame.columns.is_unique:
            raise SimulationDataError(f"{symbol}: duplicate funding column")
        times = frame["funding_time"]
        if not isinstance(times.dtype, pd.DatetimeTZDtype):
            raise SimulationDataError(f"{symbol}: funding times must be UTC-aware")
        stamps = pd.DatetimeIndex(times)
        if len(stamps) < 2 or stamps.has_duplicates or not stamps.is_monotonic_increasing:
            raise SimulationDataError(f"{symbol}: funding events unordered or duplicate")
        if not stamps.equals(stamps.tz_convert("UTC")):
            raise SimulationDataError(f"{symbol}: funding timestamps not UTC")
        if "raw_funding_time" in frame.columns:
            raw = frame["raw_funding_time"]
            if not isinstance(raw.dtype, pd.DatetimeTZDtype):
                raise SimulationDataError(f"{symbol}: raw funding time must be UTC-aware")
            raw_stamps = pd.DatetimeIndex(raw)
            if not raw_stamps.equals(raw_stamps.tz_convert("UTC")):
                raise SimulationDataError(f"{symbol}: raw funding time not UTC")
            offsets_ms = np.asarray((raw_stamps - stamps) /
                                    pd.Timedelta(milliseconds=1), dtype=float)
            if not np.isfinite(offsets_ms).all() or np.any(np.abs(offsets_ms) > 100):
                raise SimulationDataError(f"{symbol}: funding normalization exceeds 100ms")
        if stamps[0] > START or stamps[-1] != END:
            raise SimulationDataError(f"{symbol}: funding does not cover start and closeout")
        positions = grid.get_indexer(stamps)
        if np.any(positions < 0):
            raise SimulationDataError(f"{symbol}: funding not on an exact 4h mark open")
        try:
            rates = frame["rate"].to_numpy(dtype=float, na_value=np.nan)
            intervals = frame["funding_interval_hours"].to_numpy(dtype=float,
                                                                     na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise SimulationDataError(f"{symbol}: nonnumeric funding rate/interval") from exc
        if (not np.isfinite(rates).all() or not np.isfinite(intervals).all() or
                np.any(intervals <= 0) or np.any(intervals > 24)):
            raise SimulationDataError(f"{symbol}: invalid funding rate/interval")
        gaps = np.asarray((stamps[1:] - stamps[:-1]) /
                          pd.Timedelta(hours=1), dtype=float)
        if not np.allclose(gaps, intervals[1:], rtol=0, atol=1e-9):
            raise SimulationDataError(f"{symbol}: incomplete funding event sequence")
        if int(positions[0]) > int(grid.get_loc(START)):
            raise SimulationDataError(f"{symbol}: no start funding event")
        output[symbol] = dict(zip(positions.tolist(), rates.tolist(), strict=True))
    return output


def _attention_target(views: Mapping[str, pd.Series],
                      wednesday: pd.Timestamp) -> dict[str, float]:
    """Use only two *completed* Monday-Sunday windows ending three days ago."""
    if wednesday.weekday() != 2 or wednesday.hour != 0:
        raise ValueError("attention decisions are Wednesday 00:00 UTC only")
    sunday = wednesday - pd.Timedelta(days=3)
    scores: list[tuple[str, float]] = []
    for symbol in SYMBOLS:
        series = views[symbol]
        latest = float(series.loc[sunday - pd.Timedelta(days=6):sunday].sum())
        prior = float(series.loc[sunday - pd.Timedelta(days=13):
                                 sunday - pd.Timedelta(days=7)].sum())
        if latest <= 0 or prior <= 0:
            raise SimulationDataError(f"{symbol}: incomplete signal view window")
        scores.append((symbol, latest / prior - 1.0))
    low = min(scores, key=lambda pair: (pair[1], pair[0]))
    high = min(scores, key=lambda pair: (-pair[1], pair[0]))
    target = {symbol: 0.0 for symbol in SYMBOLS}
    if low[1] < 0:
        target[low[0]] = 0.5
    if high[1] > 0:
        target[high[0]] = -0.5
    return target


def run_backtest(views: Mapping[str, pd.Series],
                 trade_bars: Mapping[str, pd.DataFrame],
                 mark_bars: Mapping[str, pd.DataFrame],
                 funding_events: Mapping[str, pd.DataFrame], *,
                 slippage_rate: float = 0.001,
                 benchmark: bool = False) -> Ledger:
    """Execute the one registered weekly reversal, or the fixed passive control.

    A margin breach raises ``MarginRuinError`` at first occurrence; no five-year
    return is projected after a path that a venue might have liquidated.
    """
    if slippage_rate not in (0.001, 0.0025) or not isinstance(benchmark, bool):
        raise ValueError("only registered 10/25bp slippage and boolean control allowed")
    checked_views = _validate_views(views)
    trade = _validate_bars(trade_bars, trade=True)
    mark = _validate_bars(mark_bars, trade=False)
    grid = pd.date_range(WARMUP, END, freq="4h", tz="UTC")
    funding = _validate_funding(funding_events, grid)
    daily_volumes = {symbol: trade[symbol]["quote_volume"][:-1].reshape(-1, 6).sum(axis=1)
                     for symbol in SYMBOLS}
    start_index = int(grid.get_loc(START))
    end_index = len(grid) - 1

    cash = INITIAL_EQUITY
    holdings = {symbol: 0 for symbol in SYMBOLS}  # signed integer lot counts
    pending: dict[str, float] | None = None
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

        # Close reductions and reversals first, then open new or enlarged legs.
        for symbol in SYMBOLS:
            old, new = holdings[symbol], target_lots[symbol]
            if old and (new == 0 or old * new < 0):
                trade_lots(symbol, -old, index, stamp)
            elif old and new * old > 0 and abs(new) < abs(old):
                trade_lots(symbol, new - old, index, stamp)
        for symbol in SYMBOLS:
            old, new = holdings[symbol], target_lots[symbol]
            if new and old * new < 0:
                # A sub-minimum close remains outstanding and cannot be crossed.
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

        # Funding settles against the old signed quantity and exact mark open.
        for symbol in SYMBOLS:
            rate = funding[symbol].get(index)
            if rate is not None and holdings[symbol]:
                payment = holdings[symbol] * LOTS[symbol] * (
                    mark[symbol]["open"][index]) * rate
                cash -= payment
                funding_cost += payment
        observed_risk(index, stamp)

        if stamp == END:
            rebalance({symbol: 0.0 for symbol in SYMBOLS}, index, stamp)
        elif benchmark:
            if stamp == START or stamp.weekday() == 3 and stamp.hour == 0:
                rebalance({symbol: 0.2 for symbol in SYMBOLS}, index, stamp)
        elif stamp.weekday() == 3 and stamp.hour == 0:
            if pending is None:
                raise SimulationDataError("Thursday rebalance without Wednesday decision")
            rebalance(pending, index, stamp)
            pending = None

        after_trade = observed_risk(index, stamp)
        ordered_drawdown(after_trade, high=True, year=annual_year)
        ordered_drawdown(after_trade, high=False, year=annual_year)
        if stamp.hour == 0:
            # The comparator's first entry cost belongs to its first return.
            daily_open[day] = INITIAL_EQUITY if stamp == START else after_trade
            daily_exposed[day] = any(holdings.values())
        if stamp == END:
            break

        if stamp.year != annual_year:
            # The 00:00 settlement and trade cost finish the prior calendar
            # fold's final daily return; the new fold starts just afterwards.
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
        # Every bar is ordered favorable peak then adverse trough.  The
        # implied intrabar path is conservative, not an observed tick path.
        ordered_drawdown(high_equity, high=True, year=annual_year)
        ordered_drawdown(low_equity, high=False, year=annual_year)
        daily_high[day] = max(daily_high.get(day, -math.inf), before_funding,
                              after_trade, high_equity)
        daily_low[day] = min(daily_low.get(day, math.inf), before_funding,
                             after_trade, low_equity)

        if not benchmark and stamp.weekday() == 2 and stamp.hour == 0:
            pending = _attention_target(checked_views, stamp)

    records: list[DailyRecord] = []
    day = START
    while day < END:
        next_day = day + pd.Timedelta(days=1)
        opened = daily_open[day.date()]
        closed = daily_open[next_day.date()]
        records.append(DailyRecord(day.date(), opened, closed,
                                   closed / opened - 1.0,
                                   daily_high[day.date()],
                                   daily_low[day.date()],
                                   daily_exposed[day.date()]))
        day = next_day
    return Ledger(tuple(records), current_equity(end_index), fee_total,
                  slip_total, funding_cost, turnover, tuple(trades),
                  capacity_breaches, execution_breaches, 0, worst_drawdown,
                  annual_drawdown, not any(holdings.values()))
