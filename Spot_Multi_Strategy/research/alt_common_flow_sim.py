"""Fixed, research-only three-altcoin common-flow minute accounting.

No order routing, price fetching, optimization, or asset selection lives here.
The sole public run uses the separately committed 690a7f5 specification.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import math
from statistics import median
from typing import Iterable

from .alt_common_flow_data import AltFunding, AltMinute, SYMBOLS
from .alt_common_flow_signal import CommonFlowSignal
from .btc_eth_pairs_sim import DayRecord


UTC = timezone.utc
MINUTE = timedelta(minutes=1)
START = datetime(2021, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, tzinfo=UTC)
INPUT_START = datetime(2020, 9, 1, tzinfo=UTC)
INITIAL_EQUITY = 10_000.0
FEE = 0.0005
SCENARIOS = (
    ("base_1x", 1.0, 0.0025, False),
    ("base_2x", 2.0, 0.0025, False),
    ("doubled_cost_2x", 2.0, 0.0050, False),
    ("no_funding_credit_2x", 2.0, 0.0025, True),
)


class FlowSimulationError(ValueError):
    pass


class FlowStrategyFailure(FlowSimulationError):
    pass


@dataclass(frozen=True)
class BasketTrade:
    entry_signal_time: datetime
    entry_fill_time: datetime
    exit_fill_time: datetime
    direction: str
    regime: str
    net_pnl: float
    funding_pnl: float
    fee_pnl: float
    quantities: tuple[float, float, float]


@dataclass(frozen=True)
class SimulationResult:
    daily_records: tuple[DayRecord, ...]
    trades: tuple[BasketTrade, ...]
    max_drawdown: float
    final_equity: float
    windows_seen: int


@dataclass
class _Position:
    side: int
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    entry_exec: tuple[float, float, float]
    quantities: tuple[float, float, float]
    pre_entry_equity: float
    regime: str
    funding_pnl: float = 0.0
    fee_pnl: float = 0.0


@dataclass
class _Scenario:
    name: str
    gross: float
    adverse_rate: float
    no_funding_credit: bool
    cash: float = INITIAL_EQUITY
    position: _Position | None = None
    daily_records: list[DayRecord] = field(default_factory=list)
    trades: list[BasketTrade] = field(default_factory=list)
    peak: float = INITIAL_EQUITY
    max_drawdown: float = 0.0
    last_regular: float = INITIAL_EQUITY
    day: date | None = None
    day_start: float = INITIAL_EQUITY
    day_regular_peak: float = INITIAL_EQUITY
    day_max_regular_factor: float = 1.0
    day_min_adverse_factor: float = 1.0
    day_max_drawdown: float = 0.0

    def _record_day(self) -> None:
        if self.day is None:
            raise FlowSimulationError("no UTC day started")
        self.daily_records.append(DayRecord(
            self.day, self.day_start, self.last_regular,
            self.day_max_regular_factor, self.day_min_adverse_factor,
            self.day_max_drawdown,
        ))

    def start_day(self, day: date) -> None:
        if self.day is not None:
            self._record_day()
        if self.last_regular <= 0:
            raise FlowStrategyFailure(f"{self.name}: nonpositive day-start equity")
        self.day = day
        self.day_start = self.last_regular
        self.day_regular_peak = self.last_regular
        self.day_max_regular_factor = 1.0
        self.day_min_adverse_factor = 1.0
        self.day_max_drawdown = 0.0

    def enter(self, time: datetime, opening: tuple[float, float, float],
              side: int, signal_time: datetime, regime: str) -> None:
        if self.position is not None or self.cash <= 0 or side not in (-1, 1):
            raise FlowSimulationError("invalid simultaneous basket entry")
        pre = self.cash
        executed = tuple(p * (1 + side * self.adverse_rate) for p in opening)
        quantities = tuple(pre * self.gross / 3 / p for p in executed)
        fee = FEE * math.fsum(q * p for q, p in zip(quantities, executed))
        self.cash -= fee
        self.position = _Position(side, signal_time, time, time + 30 * MINUTE,
                                  executed, quantities, pre, regime, fee_pnl=-fee)

    def exit(self, time: datetime, opening: tuple[float, float, float]) -> None:
        pos = self.position
        if pos is None or time != pos.exit_time:
            raise FlowSimulationError("exit outside frozen thirty-minute hold")
        executed = tuple(p * (1 - pos.side * self.adverse_rate) for p in opening)
        pnl = math.fsum(pos.side * q * (p - entry)
                        for q, p, entry in zip(pos.quantities, executed, pos.entry_exec))
        fee = FEE * math.fsum(q * p for q, p in zip(pos.quantities, executed))
        self.cash += pnl - fee
        pos.fee_pnl -= fee
        self.trades.append(BasketTrade(
            pos.signal_time, pos.entry_time, time,
            "long_all" if pos.side == 1 else "short_all", pos.regime,
            self.cash - pos.pre_entry_equity, pos.funding_pnl,
            pos.fee_pnl, pos.quantities,
        ))
        self.position = None
        if self.cash <= 0:
            raise FlowStrategyFailure(f"{self.name}: nonpositive exit equity")

    def fund(self, event: AltFunding, previous_close: tuple[float, float, float]) -> None:
        pos = self.position
        if pos is None:
            return
        j = SYMBOLS.index(event.symbol)
        pnl = -pos.side * event.rate * pos.quantities[j] * previous_close[j]
        if self.no_funding_credit:
            pnl = min(0.0, pnl)
        self.cash += pnl
        pos.funding_pnl += pnl

    def mark(self, bar: AltMinute) -> None:
        pos = self.position
        if pos is None:
            regular = adverse = self.cash
        else:
            regular = self.cash + math.fsum(
                pos.side * q * (p - entry)
                for q, p, entry in zip(pos.quantities, bar.close, pos.entry_exec)
            )
            pressure = bar.low if pos.side == 1 else bar.high
            adverse = self.cash + math.fsum(
                pos.side * q * (p - entry)
                for q, p, entry in zip(pos.quantities, pressure, pos.entry_exec)
            )
        if not math.isfinite(regular + adverse) or min(regular, adverse) <= 0:
            raise FlowStrategyFailure(f"{self.name}: nonpositive minute marked equity")
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

    def finish(self, windows: int) -> SimulationResult:
        self._record_day()
        return SimulationResult(tuple(self.daily_records), tuple(self.trades),
                                self.max_drawdown, self.last_regular, windows)


class _Regime:
    """Causal five-minute closes with hourly-stride 90d trend/30d volatility."""

    def __init__(self) -> None:
        self.closes: deque[tuple[float, float, float]] = deque(maxlen=90 * 24 * 12 + 1)
        self.hour_lag: deque[tuple[float, float, float]] = deque(maxlen=13)
        self.hourly_returns: list[deque[float]] = [deque(maxlen=720) for _ in range(12)]
        self.return_sums = [0.0] * 12
        self.return_squares = [0.0] * 12
        self.daily_vol: deque[tuple[date, float]] = deque(maxlen=61)

    def _vol(self, slot: int) -> float | None:
        values = self.hourly_returns[slot]
        if len(values) < 720:
            return None
        total = self.return_sums[slot]
        variance = (self.return_squares[slot] - total * total / 720) / 719
        return math.sqrt(max(0.0, variance))

    def observe(self, time: datetime, close: tuple[float, float, float],
                *, for_signal: bool) -> str | None:
        self.closes.append(close)
        self.hour_lag.append(close)
        slot = time.minute // 5
        if len(self.hour_lag) == 13:
            old_close = self.hour_lag[0]
            ret = math.fsum(math.log(p / q) for p, q in zip(close, old_close)) / 3
            window = self.hourly_returns[slot]
            if len(window) == 720:
                old_return = window.popleft()
                self.return_sums[slot] -= old_return
                self.return_squares[slot] -= old_return * old_return
            window.append(ret)
            self.return_sums[slot] += ret
            self.return_squares[slot] += ret * ret
        if time.hour == 0 and time.minute == 0:
            vol = self._vol(slot)
            if vol is not None:
                self.daily_vol.append((time.date(), vol))
        if not for_signal:
            return None
        if len(self.closes) < 90 * 24 * 12 + 1:
            return "unknown"
        snapshots = [v for d, v in self.daily_vol if d < time.date()]
        if len(snapshots) < 60:
            return "unknown"
        current = self._vol(slot)
        if current is None or not math.isfinite(current):
            return "unknown"
        old = self.closes[0]
        trend = math.fsum(math.log(p / q) for p, q in zip(close, old)) / 3
        return ("bull" if trend >= 0 else "bear") + ("_high" if current >= median(snapshots[-60:]) else "_low")


def _funding_events(events: Iterable[AltFunding], start: datetime, end: datetime,
                    *, require_grid: bool) -> tuple[AltFunding, ...]:
    values = tuple(events)
    last: tuple[datetime, str] | None = None
    for event in values:
        if (not isinstance(event, AltFunding) or event.symbol not in SYMBOLS
                or event.time.tzinfo is None or event.time.utcoffset() != timedelta(0)
                or not start <= event.time < end or not math.isfinite(event.rate)):
            raise FlowSimulationError("invalid funding event")
        key = event.time, event.symbol
        if last is not None and key <= last:
            raise FlowSimulationError("duplicate or unsorted funding event")
        last = key
    if require_grid:
        if len(values) != (end - start).days * 9:
            raise FlowSimulationError("incomplete three-symbol eight-hour funding grid")
        for index in range(0, len(values), 3):
            anchor = start + (index // 3) * timedelta(hours=8)
            group = values[index:index + 3]
            if ({event.symbol for event in group} != set(SYMBOLS)
                    or any(not anchor <= event.time < anchor + timedelta(seconds=1)
                           for event in group)):
                raise FlowSimulationError("funding event differs from fixed UTC anchor")
    return values


def _simulate(bars: Iterable[AltMinute], funding_events: Iterable[AltFunding],
              *, input_start: datetime, start: datetime, end: datetime,
              require_funding_grid: bool) -> dict[str, SimulationResult]:
    events = _funding_events(funding_events, start, end,
                             require_grid=require_funding_grid)
    scenarios = {name: _Scenario(name, gross, adverse, no_credit)
                 for name, gross, adverse, no_credit in SCENARIOS}
    flow = CommonFlowSignal()
    regime = _Regime()
    expected = input_start
    event_index = 0
    previous_close: tuple[float, float, float] | None = None
    pending: tuple[datetime, datetime, int, str] | None = None
    exit_time: datetime | None = None
    last_entry: datetime | None = None
    for bar in bars:
        if not isinstance(bar, AltMinute) or bar.time != expected or bar.time >= end:
            raise FlowSimulationError(f"minute gap or malformed bar at expected {expected}")
        time = bar.time
        expected += MINUTE
        if time >= start and time.hour == 0 and time.minute == 0:
            for scenario in scenarios.values():
                scenario.start_day(time.date())
        if exit_time == time:
            for scenario in scenarios.values():
                scenario.exit(time, bar.opening)
            exit_time = None
        if pending is not None and pending[0] == time:
            _, signal_time, side, bucket = pending
            for scenario in scenarios.values():
                scenario.enter(time, bar.opening, side, signal_time, bucket)
            last_entry = time
            exit_time = time + 30 * MINUTE
            pending = None
        # Funding at time+milliseconds uses the previous completed minute close.
        while event_index < len(events) and events[event_index].time < time + MINUTE:
            event = events[event_index]
            if event.time < time or previous_close is None:
                raise FlowSimulationError("funding has no causal prior-minute price")
            for scenario in scenarios.values():
                scenario.fund(event, previous_close)
            event_index += 1
        if time >= start:
            for scenario in scenarios.values():
                scenario.mark(bar)
        previous_close = bar.close
        alert = flow.consume(bar)
        if (time + MINUTE).minute % 5 == 0:
            boundary = time + MINUTE
            bucket = regime.observe(boundary, bar.close, for_signal=alert is not None)
            if (alert is not None and start <= boundary < end
                    and exit_time is None and pending is None):
                entry = boundary + MINUTE
                if (entry + 30 * MINUTE < end
                        and (last_entry is None or entry >= last_entry + 60 * MINUTE)):
                    pending = (entry, boundary, alert.direction, bucket or "unknown")
    if expected != end or event_index != len(events):
        raise FlowSimulationError("incomplete formal minute or funding history")
    windows = flow.finish()
    if pending is not None or exit_time is not None:
        raise FlowSimulationError("basket remains open at OOS end")
    result = {name: scenario.finish(windows) for name, scenario in scenarios.items()}
    if any(len(item.daily_records) != (end - start).days for item in result.values()):
        raise FlowSimulationError("incomplete UTC day records")
    return result


def run_alt_flow_simulation(bars: Iterable[AltMinute],
                            funding_events: Iterable[AltFunding]) -> dict[str, SimulationResult]:
    """Run the sole frozen 2021–2025 candidate on the pinned three-coin feed."""
    return _simulate(bars, funding_events, input_start=INPUT_START, start=START,
                     end=END, require_funding_grid=True)
