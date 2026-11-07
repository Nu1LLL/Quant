"""One-shot, fee/funding-aware 4h ledger for the frozen old-coin rule.

This module has no network calls. All inputs must pass source audits before
simulation. It exposes no live-order function.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from math import floor, isfinite
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from research.old_coin_spend_source import OldCoinObservation
from research.wiki_attention_sim import DailyRecord, MarginRuinError, SimulationDataError, Trade


SYMBOLS = ("BTCUSDT", "ETHUSDT", "DOGEUSDT", "LTCUSDT", "BCHUSDT")
LOTS = {"BTCUSDT": 0.001, "ETHUSDT": 0.001, "DOGEUSDT": 1.0,
        "LTCUSDT": 0.01, "BCHUSDT": 0.01}
WARMUP = pd.Timestamp("2020-12-01 00:00", tz="UTC")
START = pd.Timestamp("2021-01-01 00:00", tz="UTC")
END = pd.Timestamp("2026-07-01 00:00", tz="UTC")
INITIAL = 10_000.0
FEE_RATE = 0.0005
MIN_NOTIONAL = 5.0
CAPACITY_RATE = 0.01
MAINTENANCE_RATE = 0.05


@dataclass(frozen=True)
class OldCoinLedger:
    records: tuple[DailyRecord, ...]
    final_equity: float
    fee_usdt: float
    slippage_usdt: float
    funding_usdt: float
    turnover_usdt: float
    trades: tuple[Trade, ...]
    capacity_breaches: int
    minimum_order_skips: int
    execution_breaches: int
    margin_proxy_breaches: int
    conservative_max_drawdown: float
    conservative_max_drawdown_by_year: dict[int, float]
    terminal_flat: bool
    capacity_schedule: tuple[tuple[date, bool], ...]


def _direction(observations: Sequence[OldCoinObservation], monday: pd.Timestamp) -> int:
    if monday.weekday() != 0 or monday.hour != 0 or monday.minute != 0:
        raise ValueError("old-coin decisions occur at Monday 00:00 UTC")
    dates = [row.observed_date for row in observations]
    if dates != sorted(set(dates)):
        raise SimulationDataError("old-coin dates unordered or duplicate")
    latest = bisect_right(dates, (monday - pd.Timedelta(days=5)).date()) - 1
    if latest < 55:
        raise SimulationDataError(f"fewer than 56 prior old-coin daily values at {monday}")
    rows = observations[latest - 55:latest + 1]
    if pd.Timestamp(rows[-1].proxy_available_at) > monday:
        raise SimulationDataError("old-coin proxy availability exceeds decision time")
    if not all(isinstance(item.spent_btc, Decimal) and item.spent_btc.is_finite()
               and item.spent_btc >= 0 for item in rows):
        raise SimulationDataError("invalid old-coin daily value")
    if any(b.observed_date != a.observed_date + timedelta(days=1)
           for a, b in zip(rows, rows[1:])):
        raise SimulationDataError("old-coin comparison has a daily date gap")
    return int(sum(item.spent_btc for item in rows[28:]) <
               sum(item.spent_btc for item in rows[:28]))


def _frames(frames: Mapping[str, pd.DataFrame], grid: pd.DatetimeIndex,
            columns: tuple[str, ...], label: str) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(frames, Mapping) or set(frames) != set(SYMBOLS):
        raise SimulationDataError(f"{label}: incorrect fixed symbol set")
    output: dict[str, dict[str, np.ndarray]] = {}
    for symbol in SYMBOLS:
        frame = frames[symbol]
        if (not isinstance(frame, pd.DataFrame) or
                not isinstance(frame.index, pd.DatetimeIndex) or
                not frame.index.equals(grid) or not frame.columns.is_unique or
                not set(columns).issubset(frame.columns)):
            raise SimulationDataError(f"{label}/{symbol}: incomplete audited 4h frame")
        try:
            data = {key: frame[key].to_numpy(dtype=float, na_value=np.nan)
                    for key in columns}
        except (TypeError, ValueError) as exc:
            raise SimulationDataError(f"{label}/{symbol}: nonnumeric field") from exc
        if (not np.isfinite(data["open"]).all() or
                np.any(data["open"] <= 0) or
                not all(np.isfinite(data[key][:-1]).all()
                        for key in ("high", "low", "close")) or
                not all(np.isnan(data[key][-1])
                        for key in ("high", "low", "close"))):
            raise SimulationDataError(f"{label}/{symbol}: invalid OHLC or terminal future")
        if "quote_volume" in data and (not np.isfinite(data["quote_volume"][:-1]).all()
                                       or np.any(data["quote_volume"][:-1] < 0)
                                       or not np.isnan(data["quote_volume"][-1])):
            raise SimulationDataError(f"{label}/{symbol}: invalid quote volume")
        output[symbol] = data
    return output


def _funding(frames: Mapping[str, pd.DataFrame],
             grid: pd.DatetimeIndex) -> dict[str, dict[int, float]]:
    if not isinstance(frames, Mapping) or set(frames) != set(SYMBOLS):
        raise SimulationDataError("funding: incorrect fixed symbol set")
    output = {}
    for symbol in SYMBOLS:
        frame = frames[symbol]
        if not isinstance(frame, pd.DataFrame) or not {
                "funding_time", "raw_funding_time", "rate", "funding_interval_hours"
        }.issubset(frame.columns):
            raise SimulationDataError(f"funding/{symbol}: missing audited columns")
        stamps = pd.DatetimeIndex(frame["funding_time"])
        raw = pd.DatetimeIndex(frame["raw_funding_time"])
        if (stamps.tz is None or str(stamps.tz) != "UTC" or
                stamps.has_duplicates or not stamps.is_monotonic_increasing or
                len(stamps) < 2 or stamps[0] != WARMUP or stamps[-1] != END or
                len(raw) != len(stamps) or raw.tz is None or str(raw.tz) != "UTC" or
                (raw < stamps).any() or
                (raw - stamps > pd.Timedelta(milliseconds=100)).any()):
            raise SimulationDataError(f"funding/{symbol}: invalid event clock")
        indices = grid.get_indexer(stamps)
        rates = frame["rate"].to_numpy(dtype=float)
        intervals = frame["funding_interval_hours"].to_numpy(dtype=float)
        gaps = np.asarray((stamps[1:] - stamps[:-1]) / pd.Timedelta(hours=1), dtype=float)
        if (np.any(indices < 0) or not np.isfinite(rates).all() or
                not np.isfinite(intervals).all() or np.any(intervals <= 0) or
                np.any(intervals > 24) or
                not np.allclose(gaps, intervals[1:], rtol=0, atol=1e-9)):
            raise SimulationDataError(f"funding/{symbol}: missing or invalid event")
        output[symbol] = dict(zip(indices.tolist(), rates.tolist(), strict=True))
    return output


def run_backtest(observations: Sequence[OldCoinObservation],
                 trade_bars: Mapping[str, pd.DataFrame],
                 mark_bars: Mapping[str, pd.DataFrame],
                 funding_events: Mapping[str, pd.DataFrame], *,
                 slippage_rate: float = 0.001,
                 latency_bars: int = 0,
                 benchmark: bool = False,
                 capacity_schedule: Mapping[date, bool] | None = None) -> OldCoinLedger:
    """Execute the registered rule, or replay its capacity schedule in stress."""
    if slippage_rate not in (0.001, 0.0025) or latency_bars not in (0, 1):
        raise ValueError("only the registered cost and latency paths are allowed")
    grid = pd.date_range(WARMUP, END, freq="4h", tz="UTC")
    trade = _frames(trade_bars, grid, ("open", "high", "low", "close", "quote_volume"), "trade")
    mark = _frames(mark_bars, grid, ("open", "high", "low", "close"), "mark")
    funding = _funding(funding_events, grid)
    daily_volumes = {symbol: trade[symbol]["quote_volume"][:-1].reshape(-1, 6).sum(axis=1)
                     for symbol in SYMBOLS}
    start_index, end_index = int(grid.get_loc(START)), len(grid) - 1
    cash = INITIAL
    holdings = {symbol: 0 for symbol in SYMBOLS}
    trades: list[Trade] = []
    fee_total = slip_total = funding_total = turnover = 0.0
    capacity_breaches = minimum_order_skips = execution_breaches = 0
    decisions: dict[date, bool] = {}
    daily_open: dict[date, float] = {}
    daily_high: dict[date, float] = {}
    daily_low: dict[date, float] = {}
    daily_exposed: dict[date, bool] = {}
    peak = INITIAL
    annual_peak = INITIAL
    annual_year = START.year
    worst_drawdown = 0.0
    annual_drawdown: dict[int, float] = {annual_year: 0.0}

    def equity(index: int) -> float:
        return cash + sum(holdings[s] * LOTS[s] * mark[s]["open"][index]
                          for s in SYMBOLS)

    def gross(index: int) -> float:
        return sum(abs(holdings[s] * LOTS[s] * mark[s]["open"][index])
                   for s in SYMBOLS)

    def risk(value: float, gross_value: float, stamp: pd.Timestamp) -> None:
        if not isfinite(value) or value <= 0 or value <= MAINTENANCE_RATE * gross_value:
            raise MarginRuinError(stamp, value, value, "old-coin account maintenance breach")

    def order(symbol: str, lots: int, index: int, stamp: pd.Timestamp) -> None:
        nonlocal cash, fee_total, slip_total, turnover, minimum_order_skips
        nonlocal execution_breaches
        if lots == 0:
            return
        quantity = lots * LOTS[symbol]
        reference = float(trade[symbol]["open"][index])
        fill = reference * (1 + slippage_rate if lots > 0 else 1 - slippage_rate)
        notional = abs(quantity * fill)
        if notional < MIN_NOTIONAL:
            minimum_order_skips += 1
            if stamp == END or holdings[symbol] == 0:
                execution_breaches += 1
            return
        fee = FEE_RATE * notional
        cash -= quantity * fill + fee
        holdings[symbol] += lots
        fee_total += fee
        slip_total += abs(quantity) * abs(fill - reference)
        turnover += notional
        prior_day = int((stamp.normalize() - WARMUP) / pd.Timedelta(days=1)) - 1
        previous_volume = daily_volumes[symbol][prior_day]
        trades.append(Trade(stamp.to_pydatetime(), symbol,
                            "buy" if lots > 0 else "sell", quantity,
                            reference, fill, notional, fee,
                            notional / previous_volume if previous_volume > 0 else float("inf")))

    def rebalance(direction: int, index: int, stamp: pd.Timestamp) -> None:
        nonlocal execution_breaches
        capital = equity(index)
        risk(capital, gross(index), stamp)
        target: dict[str, int] = {}
        for symbol in SYMBOLS:
            if direction == 0:
                target[symbol] = 0
                continue
            reference = float(trade[symbol]["open"][index])
            lots = floor(0.2 * capital / (reference * LOTS[symbol]) + 1e-12)
            if lots == 0:
                execution_breaches += 1
            target[symbol] = direction * lots
        for symbol in SYMBOLS:
            previous, wanted = holdings[symbol], target[symbol]
            if previous and (wanted == 0 or previous * wanted < 0):
                order(symbol, -previous, index, stamp)
            elif previous * wanted > 0 and abs(wanted) < abs(previous):
                order(symbol, wanted - previous, index, stamp)
        for symbol in SYMBOLS:
            previous, wanted = holdings[symbol], target[symbol]
            if wanted and previous * wanted < 0:
                execution_breaches += 1
                continue
            if wanted and (previous == 0 or previous * wanted > 0 and abs(wanted) > abs(previous)):
                order(symbol, wanted - previous, index, stamp)

    def drawdown(value: float, *, high: bool, year: int) -> None:
        nonlocal peak, annual_peak, worst_drawdown
        if high:
            peak = max(peak, value)
            annual_peak = max(annual_peak, value)
        else:
            worst_drawdown = min(worst_drawdown, value / peak - 1)
            annual_drawdown[year] = min(annual_drawdown[year], value / annual_peak - 1)

    for index in range(start_index, end_index + 1):
        stamp = grid[index]
        day = stamp.date()
        before = equity(index)
        risk(before, gross(index), stamp)
        if stamp.year != annual_year:
            annual_year = stamp.year
            annual_peak = before
            annual_drawdown[annual_year] = 0.0
        drawdown(before, high=True, year=annual_year)
        drawdown(before, high=False, year=annual_year)
        if stamp.hour == 0:
            daily_open[day] = before
            daily_exposed[day] = any(holdings.values())

        if stamp == END:
            rebalance(0, index, stamp)
            daily_open[day] = equity(index)  # final closeout cost belongs to final return
            drawdown(daily_open[day], high=False, year=annual_year)
            last_day = (stamp - pd.Timedelta(nanoseconds=1)).date()
            daily_high[last_day] = max(daily_high[last_day], before, daily_open[day])
            daily_low[last_day] = min(daily_low[last_day], before, daily_open[day])
            break

        # Funding actually occurs milliseconds after this bar starts. It
        # belongs to this UTC day and uses the completed prior 4h mark close.
        for symbol in SYMBOLS:
            rate = funding[symbol].get(index)
            if rate is not None and holdings[symbol]:
                payment = holdings[symbol] * LOTS[symbol] * mark[symbol]["close"][index - 1] * rate
                cash -= payment
                funding_total += payment
        after_funding = equity(index)
        risk(after_funding, gross(index), stamp)
        drawdown(after_funding, high=True, year=annual_year)
        drawdown(after_funding, high=False, year=annual_year)

        if stamp.weekday() == 0 and stamp.hour == 4 + 4 * latency_bars:
            monday = stamp.normalize()
            selected = 1 if benchmark else _direction(observations, monday)
            if capacity_schedule is None:
                previous_day = int((monday - WARMUP) / pd.Timedelta(days=1)) - 1
                capital = equity(index)
                allowed = True
                for symbol in SYMBOLS:
                    reference = float(trade[symbol]["open"][index])
                    intended = (selected * floor(0.2 * capital /
                                (reference * LOTS[symbol]) + 1e-12)) if selected else 0
                    adjustment = abs(intended - holdings[symbol]) * LOTS[symbol] * reference
                    if adjustment * (1 + slippage_rate) > (
                            CAPACITY_RATE * daily_volumes[symbol][previous_day]):
                        allowed = False
                if not allowed and selected != 0:
                    capacity_breaches += 1
            else:
                if monday.date() not in capacity_schedule:
                    raise SimulationDataError("missing frozen Monday capacity decision")
                allowed = capacity_schedule[monday.date()]
            active = allowed and selected != 0
            decisions[monday.date()] = active
            rebalance(selected if active else 0, index, stamp)

        after = equity(index)
        risk(after, gross(index), stamp)
        drawdown(after, high=True, year=annual_year)
        drawdown(after, high=False, year=annual_year)
        daily_exposed[day] = daily_exposed.get(day, False) or any(holdings.values())
        high_equity = low_equity = cash
        stressed_gross = 0.0
        for symbol in SYMBOLS:
            quantity = holdings[symbol] * LOTS[symbol]
            if quantity >= 0:
                high_equity += quantity * mark[symbol]["high"][index]
                low_equity += quantity * mark[symbol]["low"][index]
                stressed_gross += quantity * mark[symbol]["low"][index]
            else:
                high_equity += quantity * mark[symbol]["low"][index]
                low_equity += quantity * mark[symbol]["high"][index]
                stressed_gross += abs(quantity) * mark[symbol]["high"][index]
        risk(low_equity, stressed_gross, stamp)
        drawdown(high_equity, high=True, year=annual_year)
        drawdown(low_equity, high=False, year=annual_year)
        daily_high[day] = max(daily_high.get(day, float("-inf")), before,
                              after_funding, after, high_equity)
        daily_low[day] = min(daily_low.get(day, float("inf")), before,
                             after_funding, after, low_equity)

    records = []
    day = START
    while day < END:
        opened = daily_open[day.date()]
        closed = daily_open[(day + pd.Timedelta(days=1)).date()]
        records.append(DailyRecord(day.date(), opened, closed, closed / opened - 1,
                                   daily_high[day.date()], daily_low[day.date()],
                                   daily_exposed[day.date()]))
        day += pd.Timedelta(days=1)
    return OldCoinLedger(tuple(records), equity(end_index), fee_total, slip_total,
                      funding_total, turnover, tuple(trades), capacity_breaches,
                      minimum_order_skips, execution_breaches, 0, worst_drawdown,
                      annual_drawdown, not any(holdings.values()),
                      tuple(sorted(decisions.items())))
