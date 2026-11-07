"""Frozen hourly, monthly funding-spread simulation for `991a4ae`.

It consumes the pinned feed, never fetches data or routes a trade, and has no
runtime asset, threshold, leverage, or cost optimization parameters.
"""

from __future__ import annotations

from calendar import monthrange
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import math
from statistics import median
from typing import Iterable

from .funding_spread_data import FundingEvent, HourBar, SYMBOLS, TerminalOpen
from .btc_eth_pairs_sim import DayRecord


UTC = timezone.utc
HOUR = timedelta(hours=1)
START = datetime(2021, 1, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, 1, tzinfo=UTC)
INPUT_START = datetime(2020, 12, 1, tzinfo=UTC)
EQUITY = 10_000.0
FEE = 0.0005
MIN_CARRY_AFTER_COST = 0.006
SCENARIOS = (
    ("base_1x", 1.0, 0.0025, False),
    ("base_2x", 2.0, 0.0025, False),
    ("doubled_cost_2x", 2.0, 0.0050, False),
    ("no_funding_credit_2x", 2.0, 0.0025, True),
)


class FundingSimulationError(ValueError):
    pass


class FundingStrategyFailure(FundingSimulationError):
    pass


@dataclass(frozen=True)
class SpreadTrade:
    entry_signal_time: datetime
    entry_fill_time: datetime
    exit_fill_time: datetime
    direction: str
    regime: str
    net_pnl: float
    funding_pnl: float
    fee_pnl: float
    long_symbol: str
    short_symbol: str
    long_quantity: float
    short_quantity: float


@dataclass(frozen=True)
class SimulationResult:
    daily_records: tuple[DayRecord, ...]
    trades: tuple[SpreadTrade, ...]
    max_drawdown: float
    final_equity: float
    decision_opportunities: int


@dataclass
class _Position:
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    long_index: int
    short_index: int
    long_exec: float
    short_exec: float
    long_quantity: float
    short_quantity: float
    pre_equity: float
    regime: str
    funding_pnl: float = 0.0
    fee_pnl: float = 0.0


@dataclass
class _Scenario:
    name: str
    gross: float
    adverse_rate: float
    no_funding_credit: bool
    cash: float = EQUITY
    position: _Position | None = None
    daily_records: list[DayRecord] = field(default_factory=list)
    trades: list[SpreadTrade] = field(default_factory=list)
    peak: float = EQUITY
    max_drawdown: float = 0.0
    last_regular: float = EQUITY
    day: date | None = None
    day_start: float = EQUITY
    day_regular_peak: float = EQUITY
    day_max_regular_factor: float = 1.0
    day_min_adverse_factor: float = 1.0
    day_max_drawdown: float = 0.0

    def _account(self, regular: float, adverse: float) -> None:
        if not math.isfinite(regular + adverse) or min(regular, adverse) <= 0:
            raise FundingStrategyFailure(f"{self.name}: nonpositive hourly equity")
        self.last_regular = regular
        self.peak = max(self.peak, regular)
        self.max_drawdown = max(self.max_drawdown, 1 - adverse / self.peak)
        if self.day is not None:
            self.day_regular_peak = max(self.day_regular_peak, regular)
            self.day_max_regular_factor = max(self.day_max_regular_factor,
                                              regular / self.day_start)
            self.day_min_adverse_factor = min(self.day_min_adverse_factor,
                                              adverse / self.day_start)
            self.day_max_drawdown = max(self.day_max_drawdown,
                                        1 - adverse / self.day_regular_peak)

    def _equity(self, prices: tuple[float, float, float]) -> float:
        pos = self.position
        if pos is None:
            return self.cash
        return (self.cash + pos.long_quantity * (prices[pos.long_index] - pos.long_exec)
                + pos.short_quantity * (pos.short_exec - prices[pos.short_index]))

    def boundary(self, opening: tuple[float, float, float]) -> None:
        regular = self._equity(opening)
        self._account(regular, regular)

    def mark(self, bar: HourBar) -> None:
        regular = self._equity(bar.close)
        pos = self.position
        if pos is None:
            adverse = regular
        else:
            adverse = (self.cash
                       + pos.long_quantity * (bar.low[pos.long_index] - pos.long_exec)
                       + pos.short_quantity * (pos.short_exec - bar.high[pos.short_index]))
        self._account(regular, adverse)

    def _record_day(self) -> None:
        if self.day is None:
            raise FundingSimulationError("no UTC 01:00 day started")
        self.daily_records.append(DayRecord(
            self.day, self.day_start, self.last_regular,
            self.day_max_regular_factor, self.day_min_adverse_factor,
            self.day_max_drawdown,
        ))

    def start_day(self, day: date) -> None:
        if self.day is not None:
            self._record_day()
        self.day = day
        self.day_start = self.last_regular
        self.day_regular_peak = self.last_regular
        self.day_max_regular_factor = 1.0
        self.day_min_adverse_factor = 1.0
        self.day_max_drawdown = 0.0

    def finish_day(self) -> None:
        self._record_day()

    def enter(self, time: datetime, opening: tuple[float, float, float],
              signal_time: datetime, long_index: int, short_index: int,
              exit_time: datetime, regime: str) -> None:
        if self.position is not None or long_index == short_index or self.cash <= 0:
            raise FundingSimulationError("invalid funding-spread entry")
        pre = self.cash
        long_exec = opening[long_index] * (1 + self.adverse_rate)
        short_exec = opening[short_index] * (1 - self.adverse_rate)
        long_quantity = pre * self.gross / 2 / long_exec
        short_quantity = pre * self.gross / 2 / short_exec
        fee = FEE * (long_quantity * long_exec + short_quantity * short_exec)
        self.cash -= fee
        self.position = _Position(signal_time, time, exit_time,
                                  long_index, short_index, long_exec, short_exec,
                                  long_quantity, short_quantity, pre, regime,
                                  fee_pnl=-fee)

    def exit(self, time: datetime, opening: tuple[float, float, float]) -> None:
        pos = self.position
        if pos is None or time != pos.exit_time:
            raise FundingSimulationError("exit outside fixed monthly boundary")
        long_exec = opening[pos.long_index] * (1 - self.adverse_rate)
        short_exec = opening[pos.short_index] * (1 + self.adverse_rate)
        pnl = (pos.long_quantity * (long_exec - pos.long_exec)
               + pos.short_quantity * (pos.short_exec - short_exec))
        fee = FEE * (pos.long_quantity * long_exec + pos.short_quantity * short_exec)
        self.cash += pnl - fee
        pos.fee_pnl -= fee
        self.trades.append(SpreadTrade(
            pos.signal_time, pos.entry_time, time, "long_low_short_high",
            pos.regime, self.cash - pos.pre_equity, pos.funding_pnl,
            pos.fee_pnl, SYMBOLS[pos.long_index], SYMBOLS[pos.short_index],
            pos.long_quantity, pos.short_quantity,
        ))
        self.position = None
        if self.cash <= 0:
            raise FundingStrategyFailure(f"{self.name}: nonpositive exit cash")

    def fund(self, event: FundingEvent,
             previous_close: tuple[float, float, float]) -> None:
        pos = self.position
        if pos is None:
            return
        index = SYMBOLS.index(event.symbol)
        if index == pos.long_index:
            pnl = -event.rate * pos.long_quantity * previous_close[index]
        elif index == pos.short_index:
            pnl = event.rate * pos.short_quantity * previous_close[index]
        else:
            return
        if self.no_funding_credit:
            pnl = min(0.0, pnl)
        self.cash += pnl
        pos.funding_pnl += pnl

    def result(self, opportunities: int) -> SimulationResult:
        return SimulationResult(tuple(self.daily_records), tuple(self.trades),
                                self.max_drawdown, self.last_regular, opportunities)


class _Regime:
    def __init__(self) -> None:
        self.closes: deque[tuple[float, float, float]] = deque(maxlen=721)
        self.returns: deque[float] = deque(maxlen=336)
        self.total = 0.0
        self.squares = 0.0
        self.snapshots: deque[tuple[date, float]] = deque(maxlen=31)

    def _sigma(self) -> float | None:
        if len(self.returns) < 336:
            return None
        variance = (self.squares - self.total * self.total / 336) / 335
        return math.sqrt(max(0.0, variance))

    def observe(self, boundary: datetime, close: tuple[float, float, float],
                *, decision: bool) -> str | None:
        old = self.closes[-1] if self.closes else None
        self.closes.append(close)
        if old is not None:
            value = math.fsum(math.log(p / q) for p, q in zip(close, old)) / 3
            if len(self.returns) == 336:
                removed = self.returns.popleft()
                self.total -= removed
                self.squares -= removed * removed
            self.returns.append(value)
            self.total += value
            self.squares += value * value
        if boundary.hour == 17:
            sigma = self._sigma()
            if sigma is not None:
                self.snapshots.append((boundary.date(), sigma))
        if not decision:
            return None
        earlier = [v for d, v in self.snapshots if d < boundary.date()]
        if len(self.closes) != 721 or len(earlier) != 30:
            return "unknown"
        sigma = self._sigma()
        if sigma is None or not math.isfinite(sigma):
            return "unknown"
        trend = math.fsum(math.log(p / q) for p, q in zip(close, self.closes[0])) / 3
        return ("bull" if trend >= 0 else "bear") + ("_high" if sigma >= median(earlier) else "_low")


def _next_month(day: date) -> date:
    return date(day.year + (day.month == 12), day.month % 12 + 1, 1)


def _events(funding: Iterable[FundingEvent]) -> tuple[FundingEvent, ...]:
    events = tuple(funding)
    first = datetime(2021, 1, 1, tzinfo=UTC)
    anchors = int((END - first).total_seconds() // (8 * 3600)) + 1
    if len(events) != anchors * 3:
        raise FundingSimulationError("funding grid has missing or extra events")
    prior: tuple[datetime, str] | None = None
    for index, event in enumerate(events):
        if (not isinstance(event, FundingEvent) or event.symbol not in SYMBOLS
                or event.time.tzinfo is None or event.time.utcoffset() != timedelta(0)
                or not math.isfinite(event.rate)):
            raise FundingSimulationError("malformed funding event")
        key = event.time, event.symbol
        if prior is not None and key <= prior:
            raise FundingSimulationError("funding events are unsorted or duplicated")
        prior = key
    for i in range(0, len(events), 3):
        anchor = first + i // 3 * timedelta(hours=8)
        group = events[i:i + 3]
        if ({event.symbol for event in group} != set(SYMBOLS)
                or any(not anchor <= event.time < anchor + timedelta(seconds=1)
                       for event in group)):
            raise FundingSimulationError("funding event off its eight-hour anchor")
    return events


@dataclass(frozen=True)
class _Decision:
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    long_index: int
    short_index: int
    regime: str


def run_funding_spread_simulation(
    bars: Iterable[HourBar | TerminalOpen], funding: Iterable[FundingEvent],
) -> dict[str, SimulationResult]:
    events = _events(funding)
    scenarios = {name: _Scenario(name, gross, adverse, no_credit)
                 for name, gross, adverse, no_credit in SCENARIOS}
    history = {symbol: deque(maxlen=90) for symbol in SYMBOLS}
    regime = _Regime()
    expected = INPUT_START
    previous_close: tuple[float, float, float] | None = None
    event_index = 0
    pending: _Decision | None = None
    opportunities = 0
    terminal_seen = False
    for bar in bars:
        if bar.time != expected:
            raise FundingSimulationError(f"hourly gap or duplicate at {expected}")
        if isinstance(bar, TerminalOpen):
            if bar.time != END or terminal_seen:
                raise FundingSimulationError("terminal open is off the fixed end")
            for scenario in scenarios.values():
                if scenario.position is not None:
                    scenario.exit(bar.time, bar.opening)
                scenario.boundary(bar.opening)
                scenario.finish_day()
            terminal_seen = True
            expected += HOUR
            continue
        if not isinstance(bar, HourBar) or bar.time >= END:
            raise FundingSimulationError("invalid hourly bar")
        time = bar.time
        expected += HOUR
        if START <= time and time.hour == 1:
            for scenario in scenarios.values():
                if scenario.position is not None and scenario.position.exit_time == time:
                    scenario.exit(time, bar.opening)
                scenario.boundary(bar.opening)
                scenario.start_day(time.date())
            if pending is not None and pending.entry_time == time:
                for scenario in scenarios.values():
                    scenario.enter(time, bar.opening, pending.signal_time,
                                   pending.long_index, pending.short_index,
                                   pending.exit_time, pending.regime)
                pending = None
        while event_index < len(events) and events[event_index].time < time + HOUR:
            event = events[event_index]
            if event.time < time or previous_close is None:
                raise FundingSimulationError("funding has no completed prior-hour price")
            history[event.symbol].append(event.rate)
            for scenario in scenarios.values():
                scenario.fund(event, previous_close)
            event_index += 1
        if time >= START:
            for scenario in scenarios.values():
                scenario.mark(bar)
        previous_close = bar.close
        boundary = time + HOUR
        decision = (boundary.hour == 17
                    and boundary.date().day == monthrange(boundary.year, boundary.month)[1]
                    and date(2021, 1, 1) <= boundary.date() <= date(2025, 11, 30))
        bucket = regime.observe(boundary, bar.close, decision=decision)
        if decision:
            opportunities += 1
            if pending is not None or any(len(history[s]) != 90 for s in SYMBOLS):
                raise FundingSimulationError("decision lacks ninety completed funding rates")
            means = {s: math.fsum(history[s]) / 90 for s in SYMBOLS}
            ranking = sorted(SYMBOLS, key=lambda s: (means[s], s))
            low, high = ranking[0], ranking[-1]
            following = _next_month(boundary.date())
            after = _next_month(following)
            n_settlements = 3 * monthrange(following.year, following.month)[1]
            predicted = .5 * n_settlements * (means[high] - means[low])
            if predicted > MIN_CARRY_AFTER_COST:
                pending = _Decision(boundary,
                                    datetime(following.year, following.month, 1, 1, tzinfo=UTC),
                                    datetime(after.year, after.month, 1, 1, tzinfo=UTC),
                                    SYMBOLS.index(low), SYMBOLS.index(high), bucket or "unknown")
    if not terminal_seen or event_index != len(events) or opportunities != 59:
        raise FundingSimulationError("incomplete fixed hourly, funding, or decision path")
    if pending is not None or any(s.position is not None for s in scenarios.values()):
        raise FundingSimulationError("nonflat account at the terminal boundary")
    if any(len(s.daily_records) != 1826 for s in scenarios.values()):
        raise FundingSimulationError("incomplete 01:00-anchored daily path")
    return {name: scenario.result(opportunities) for name, scenario in scenarios.items()}
