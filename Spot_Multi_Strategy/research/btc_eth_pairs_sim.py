"""Frozen, research-only BTC/ETH USD-M distance-pairs minute simulator.

The public entry point has no tunable market, timing, cost or leverage inputs.
It consumes validated, synchronized minute bars and actual funding events, and
keeps only hourly formation history, open positions and daily path summaries.
It neither fetches data nor sends orders.  See the separately committed
2026-11-07 BTC/ETH distance-pairs pre-registration for the frozen rules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import math
from statistics import median, stdev
from typing import Iterable


UTC = timezone.utc
MINUTE = timedelta(minutes=1)
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
INITIAL_EQUITY = 10_000.0
FEE_RATE = 0.0005
BASE_ADVERSE_RATE = 0.0025
STRESS_ADVERSE_RATE = 0.0050
SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT"})


class PairSimulationDataError(ValueError):
    """Incomplete or malformed inputs invalidate the entire formal run."""


class SignalQualityFailure(ValueError):
    """A frozen formation block cannot produce a finite standardized score."""


class StrategyFailure(ValueError):
    """A predetermined scenario reached nonpositive marked equity."""


@dataclass(frozen=True)
class DayRecord:
    date: date
    equity_start: float
    equity_end: float
    max_regular_factor: float
    min_adverse_factor: float
    max_intraday_drawdown: float


@dataclass(frozen=True)
class PairTrade:
    entry_signal_time: datetime
    entry_fill_time: datetime
    exit_fill_time: datetime
    direction: str
    regime: str
    net_pnl: float
    funding_pnl: float
    fee_pnl: float
    btc_quantity: float
    eth_quantity: float


@dataclass(frozen=True)
class SimulationResult:
    daily_records: tuple[DayRecord, ...]
    trades: tuple[PairTrade, ...]
    max_drawdown: float
    final_equity: float
    completed_cycles: int


@dataclass(frozen=True)
class _Spec:
    input_start: datetime
    start: datetime
    end: datetime
    formation_hours: int
    trading_hours: int
    exact_funding_grid: bool

    def __post_init__(self) -> None:
        for value in (self.input_start, self.start, self.end):
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError("simulation bounds must be UTC")
            if value.minute or value.second or value.microsecond:
                raise ValueError("simulation bounds must be hourly")
        # The first close in a minute-open feed arrives one hour after its
        # first bar; N formation returns require N+1 completed closes.
        if not self.input_start <= self.start - (self.formation_hours + 1) * HOUR:
            raise ValueError("formation history is unavailable")
        if not self.input_start < self.start < self.end:
            raise ValueError("invalid simulation period")
        if self.formation_hours < 2 or self.trading_hours < 2:
            raise ValueError("invalid cycle length")
        if (self.end - self.start) % DAY:
            raise ValueError("OOS must contain whole UTC days")


_FORMAL_SPEC = _Spec(
    datetime(2020, 1, 1, tzinfo=UTC),
    datetime(2021, 1, 1, tzinfo=UTC),
    datetime(2026, 1, 1, tzinfo=UTC),
    480, 240, True,
)


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PairSimulationDataError(f"{label} must be a timezone-aware datetime")
    if value.utcoffset() != timedelta(0):
        raise PairSimulationDataError(f"{label} must be UTC")
    return value.astimezone(UTC)


def _price(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PairSimulationDataError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise PairSimulationDataError(f"{label} must be finite and positive")
    return result


def _bar_values(bar: object) -> tuple[datetime, tuple[float, ...]]:
    try:
        time = _utc(getattr(bar, "time"), "minute time")
        raw = tuple(getattr(bar, key) for key in (
            "btc_open", "btc_high", "btc_low", "btc_close",
            "eth_open", "eth_high", "eth_low", "eth_close",
        ))
    except AttributeError as exc:
        raise PairSimulationDataError("paired minute bar is missing a field") from exc
    if time.second or time.microsecond:
        raise PairSimulationDataError("minute bar is off the UTC minute grid")
    values = tuple(_price(value, "minute OHLC") for value in raw)
    for offset in (0, 4):
        opening, high, low, close = values[offset:offset + 4]
        if low > min(opening, close) or high < max(opening, close):
            raise PairSimulationDataError("inconsistent minute OHLC")
    return time, values


@dataclass(frozen=True)
class _Funding:
    time: datetime
    symbol: str
    rate: float


def _funding_events(events: Iterable[object], spec: _Spec) -> list[_Funding]:
    result: list[_Funding] = []
    previous: tuple[datetime, str] | None = None
    for event in events:
        try:
            time = _utc(getattr(event, "time"), "funding time")
            symbol = getattr(event, "symbol")
            rate = float(getattr(event, "rate"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise PairSimulationDataError("malformed funding event") from exc
        if symbol not in SYMBOLS or not math.isfinite(rate):
            raise PairSimulationDataError("invalid funding symbol or rate")
        if not spec.start <= time < spec.end or time.microsecond % 1000:
            raise PairSimulationDataError("funding event outside OOS or not millisecond aligned")
        key = time, symbol
        if previous is not None and key <= previous:
            raise PairSimulationDataError("funding events are unsorted or duplicate")
        previous = key
        result.append(_Funding(time, symbol, rate))
    if spec.exact_funding_grid:
        anchor = spec.start
        index = 0
        while anchor < spec.end:
            pair = result[index:index + 2]
            if len(pair) != 2 or {item.symbol for item in pair} != SYMBOLS:
                raise PairSimulationDataError("funding anchor lacks one event per leg")
            if any(not anchor <= item.time < anchor + timedelta(seconds=1)
                   for item in pair):
                raise PairSimulationDataError("funding event is off the fixed eight-hour grid")
            index += 2
            anchor += timedelta(hours=8)
        if index != len(result):
            raise PairSimulationDataError("extra funding events")
    return result


class _RollingStd:
    def __init__(self, width: int) -> None:
        self.width = width
        self.values: deque[float] = deque()
        self.total = 0.0
        self.squares = 0.0

    def append(self, value: float) -> None:
        if len(self.values) == self.width:
            old = self.values.popleft()
            self.total -= old
            self.squares -= old * old
        self.values.append(value)
        self.total += value
        self.squares += value * value

    def sigma(self) -> float | None:
        if len(self.values) < self.width:
            return None
        variance = (self.squares - self.total * self.total / self.width) / (
            self.width - 1
        )
        return math.sqrt(max(0.0, variance))


class _HourlyState:
    def __init__(self, spec: _Spec) -> None:
        self.spec = spec
        self.closes: deque[tuple[float, float]] = deque(maxlen=spec.formation_hours + 1)
        self.btc_90d_closes: deque[float] = deque(maxlen=90 * 24 + 1)
        self.btc_vol = _RollingStd(30 * 24)
        self.daily_vol: deque[tuple[date, float]] = deque(maxlen=366)
        self.cycle_end: datetime | None = None
        self.mu_btc = self.sd_btc = self.mu_eth = self.sd_eth = 0.0
        self.mu_distance = self.sd_distance = 0.0
        self.index_btc = self.index_eth = 1.0
        self.previous_z = 0.0
        self.completed_cycles = 0
        self.full_cycle_end = spec.start + (
            (spec.end - spec.start) // (spec.trading_hours * HOUR)
        ) * spec.trading_hours * HOUR

    def _start_cycle(self, time: datetime) -> None:
        n = self.spec.formation_hours
        if len(self.closes) != n + 1:
            raise PairSimulationDataError("insufficient hourly formation closes")
        closes = list(self.closes)
        btc = [math.log(closes[j][0] / closes[j - 1][0]) for j in range(1, n + 1)]
        eth = [math.log(closes[j][1] / closes[j - 1][1]) for j in range(1, n + 1)]
        self.mu_btc = math.fsum(btc) / n
        self.mu_eth = math.fsum(eth) / n
        self.sd_btc = stdev(btc)
        self.sd_eth = stdev(eth)
        if not (math.isfinite(self.sd_btc) and math.isfinite(self.sd_eth)
                and self.sd_btc > 0 and self.sd_eth > 0):
            raise SignalQualityFailure(f"zero or invalid leg formation sigma at {time}")
        self.index_btc = self.index_eth = 1.0
        distances: list[float] = []
        for r_btc, r_eth in zip(btc, eth):
            self.index_btc += (r_btc - self.mu_btc) / self.sd_btc
            self.index_eth += (r_eth - self.mu_eth) / self.sd_eth
            distances.append(self.index_btc - self.index_eth)
        self.mu_distance = math.fsum(distances) / n
        self.sd_distance = stdev(distances)
        if not math.isfinite(self.sd_distance) or self.sd_distance <= 0:
            raise SignalQualityFailure(f"zero or invalid distance sigma at {time}")
        self.previous_z = (distances[-1] - self.mu_distance) / self.sd_distance
        self.cycle_end = time + self.spec.trading_hours * HOUR
        self.completed_cycles += 1

    def _regime(self, time: datetime, close_btc: float) -> str:
        sigma = self.btc_vol.sigma()
        past = [value for day, value in self.daily_vol if day < time.date()][-365:]
        if len(self.btc_90d_closes) < 90 * 24 + 1 or sigma is None or len(past) < 300:
            return "unknown"
        trend = "bull" if close_btc >= self.btc_90d_closes[0] else "bear"
        volatility = "high" if sigma >= median(past) else "low"
        return f"{trend}_{volatility}"

    def observe(self, time: datetime, close_btc: float, close_eth: float
                ) -> tuple[float, float, str] | None:
        previous = self.closes[-1] if self.closes else None
        self.closes.append((close_btc, close_eth))
        self.btc_90d_closes.append(close_btc)
        if previous is not None:
            self.btc_vol.append(math.log(close_btc / previous[0]))
        if time.hour == 0 and time.minute == 0:
            sigma = self.btc_vol.sigma()
            if sigma is not None:
                self.daily_vol.append((time.date(), sigma))
        if not self.spec.start <= time < self.full_cycle_end:
            return None
        if (time - self.spec.start) % (self.spec.trading_hours * HOUR) == timedelta(0):
            self._start_cycle(time)
            return None
        if self.cycle_end is None or time >= self.cycle_end or previous is None:
            raise PairSimulationDataError("hourly score outside a formed cycle")
        r_btc = math.log(close_btc / previous[0])
        r_eth = math.log(close_eth / previous[1])
        self.index_btc += (r_btc - self.mu_btc) / self.sd_btc
        self.index_eth += (r_eth - self.mu_eth) / self.sd_eth
        z = ((self.index_btc - self.index_eth) - self.mu_distance) / self.sd_distance
        if not math.isfinite(z):
            raise SignalQualityFailure("nonfinite hourly pair score")
        before = self.previous_z
        self.previous_z = z
        return before, z, self._regime(time, close_btc)


@dataclass
class _Position:
    side_btc: int
    entry_signal_time: datetime
    entry_fill_time: datetime
    entry_btc: float
    entry_eth: float
    quantity_btc: float
    quantity_eth: float
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
    trades: list[PairTrade] = field(default_factory=list)
    peak: float = INITIAL_EQUITY
    max_drawdown: float = 0.0
    last_regular: float = INITIAL_EQUITY
    day: date | None = None
    day_start: float = INITIAL_EQUITY
    day_peak_regular: float = INITIAL_EQUITY
    day_max_regular_factor: float = 1.0
    day_min_adverse_factor: float = 1.0
    day_max_drawdown: float = 0.0

    def start_day(self, day: date) -> None:
        if self.day is not None:
            self.daily_records.append(DayRecord(
                self.day, self.day_start, self.last_regular,
                self.day_max_regular_factor, self.day_min_adverse_factor,
                self.day_max_drawdown,
            ))
        if self.last_regular <= 0:
            raise StrategyFailure(f"{self.name}: nonpositive UTC day-start equity")
        self.day = day
        self.day_start = self.last_regular
        self.day_peak_regular = self.last_regular
        self.day_max_regular_factor = 1.0
        self.day_min_adverse_factor = 1.0
        self.day_max_drawdown = 0.0

    def finish_day(self) -> None:
        if self.day is None:
            raise PairSimulationDataError("missing first OOS day")
        self.daily_records.append(DayRecord(
            self.day, self.day_start, self.last_regular,
            self.day_max_regular_factor, self.day_min_adverse_factor,
            self.day_max_drawdown,
        ))

    def enter(self, time: datetime, btc_open: float, eth_open: float,
              side_btc: int, signal_time: datetime, regime: str) -> None:
        if self.position is not None or self.cash <= 0:
            raise StrategyFailure(f"{self.name}: cannot enter with open or bankrupt account")
        pre_entry = self.cash
        btc_exec = btc_open * (1 + side_btc * self.adverse_rate)
        eth_exec = eth_open * (1 - side_btc * self.adverse_rate)
        quantity_btc = pre_entry * self.gross / 2 / btc_exec
        quantity_eth = pre_entry * self.gross / 2 / eth_exec
        fee = FEE_RATE * (quantity_btc * btc_exec + quantity_eth * eth_exec)
        self.cash -= fee
        self.position = _Position(
            side_btc, signal_time, time, btc_exec, eth_exec,
            quantity_btc, quantity_eth, pre_entry, regime, fee_pnl=-fee,
        )

    def exit(self, time: datetime, btc_open: float, eth_open: float) -> None:
        pos = self.position
        if pos is None:
            raise PairSimulationDataError("exit without an open pair")
        btc_exec = btc_open * (1 - pos.side_btc * self.adverse_rate)
        eth_exec = eth_open * (1 + pos.side_btc * self.adverse_rate)
        pnl_btc = pos.side_btc * pos.quantity_btc * (btc_exec - pos.entry_btc)
        pnl_eth = -pos.side_btc * pos.quantity_eth * (eth_exec - pos.entry_eth)
        fee = FEE_RATE * (pos.quantity_btc * btc_exec + pos.quantity_eth * eth_exec)
        self.cash += pnl_btc + pnl_eth - fee
        pos.fee_pnl -= fee
        self.trades.append(PairTrade(
            pos.entry_signal_time, pos.entry_fill_time, time,
            "long_btc_short_eth" if pos.side_btc == 1 else "short_btc_long_eth",
            pos.regime, self.cash - pos.pre_entry_equity, pos.funding_pnl,
            pos.fee_pnl, pos.quantity_btc, pos.quantity_eth,
        ))
        self.position = None
        if self.cash <= 0:
            raise StrategyFailure(f"{self.name}: nonpositive equity at pair exit")

    def fund(self, symbol: str, rate: float, btc_proxy: float, eth_proxy: float) -> None:
        pos = self.position
        if pos is None:
            return
        if symbol == "BTCUSDT":
            pnl = -pos.side_btc * rate * pos.quantity_btc * btc_proxy
        else:
            pnl = pos.side_btc * rate * pos.quantity_eth * eth_proxy
        if self.no_funding_credit:
            pnl = min(0.0, pnl)
        self.cash += pnl
        pos.funding_pnl += pnl

    def mark(self, values: tuple[float, ...]) -> None:
        pos = self.position
        if pos is None:
            regular = adverse = self.cash
        else:
            side = pos.side_btc
            regular = self.cash + side * pos.quantity_btc * (values[3] - pos.entry_btc)
            regular -= side * pos.quantity_eth * (values[7] - pos.entry_eth)
            btc_adverse = values[2] if side == 1 else values[1]
            eth_adverse = values[5] if side == 1 else values[6]
            adverse = self.cash + side * pos.quantity_btc * (btc_adverse - pos.entry_btc)
            adverse -= side * pos.quantity_eth * (eth_adverse - pos.entry_eth)
        if min(regular, adverse) <= 0 or not math.isfinite(regular + adverse):
            raise StrategyFailure(f"{self.name}: nonpositive minute marked equity")
        self.last_regular = regular
        self.peak = max(self.peak, regular)
        self.max_drawdown = max(self.max_drawdown, 1 - min(regular, adverse) / self.peak)
        if self.day is not None:
            self.day_peak_regular = max(self.day_peak_regular, regular)
            self.day_max_regular_factor = max(self.day_max_regular_factor,
                                              regular / self.day_start)
            self.day_min_adverse_factor = min(self.day_min_adverse_factor,
                                              adverse / self.day_start)
            self.day_max_drawdown = max(self.day_max_drawdown,
                                        1 - adverse / self.day_peak_regular)

    def result(self, completed_cycles: int) -> SimulationResult:
        return SimulationResult(
            tuple(self.daily_records), tuple(self.trades),
            self.max_drawdown, self.last_regular, completed_cycles,
        )


@dataclass(frozen=True)
class _Pending:
    kind: str
    time: datetime
    signal_time: datetime
    side_btc: int
    regime: str


def _simulate(bars: Iterable[object], funding_events: Iterable[object],
              *, spec: _Spec) -> dict[str, SimulationResult]:
    events = _funding_events(funding_events, spec)
    scenarios = {
        "base_1x": _Scenario("base_1x", 1.0, BASE_ADVERSE_RATE, False),
        "base_2x": _Scenario("base_2x", 2.0, BASE_ADVERSE_RATE, False),
        "doubled_cost_2x": _Scenario("doubled_cost_2x", 2.0, STRESS_ADVERSE_RATE, False),
        "no_funding_credit_2x": _Scenario("no_funding_credit_2x", 2.0,
                                           BASE_ADVERSE_RATE, True),
    }
    signal = _HourlyState(spec)
    expected_time = spec.input_start
    event_index = 0
    previous_close: tuple[float, float] | None = None
    pending: _Pending | None = None
    side_btc = 0
    forced_exit: datetime | None = None
    for raw_bar in bars:
        time, values = _bar_values(raw_bar)
        if time != expected_time or time >= spec.end:
            raise PairSimulationDataError(
                f"minute grid mismatch at {time}; expected {expected_time}"
            )
        expected_time += MINUTE
        if time >= spec.start and time.hour == 0 and time.minute == 0:
            for scenario in scenarios.values():
                scenario.start_day(time.date())
        # Scheduled market orders are filled at this minute's reference open.
        # At an identical funding timestamp, entry is inclusive and exit exclusive.
        if pending is not None and pending.time == time:
            if pending.kind == "entry":
                for scenario in scenarios.values():
                    scenario.enter(time, values[0], values[4], pending.side_btc,
                                   pending.signal_time, pending.regime)
                side_btc = pending.side_btc
            else:
                for scenario in scenarios.values():
                    scenario.exit(time, values[0], values[4])
                side_btc = 0
                forced_exit = None
            pending = None
        elif side_btc and forced_exit == time:
            for scenario in scenarios.values():
                scenario.exit(time, values[0], values[4])
            side_btc = 0
            forced_exit = None
        while event_index < len(events) and events[event_index].time < time + MINUTE:
            event = events[event_index]
            if event.time < time or previous_close is None:
                raise PairSimulationDataError("funding lacks completed prior minute close")
            for scenario in scenarios.values():
                scenario.fund(event.symbol, event.rate,
                              previous_close[0], previous_close[1])
            event_index += 1
        if time >= spec.start:
            for scenario in scenarios.values():
                scenario.mark(values)
        previous_close = values[3], values[7]
        # Minute 59 closes the completed UTC hour at the next hour boundary.
        if time.minute == 59:
            boundary = time + MINUTE
            score = signal.observe(boundary, values[3], values[7])
            if score is not None:
                before, now, regime = score
                if side_btc == 0 and pending is None:
                    new_side = 0
                    if before < 2 <= now:
                        new_side = -1
                    elif before > -2 >= now:
                        new_side = 1
                    fill_time = boundary + HOUR + MINUTE
                    if new_side and signal.cycle_end is not None and fill_time < signal.cycle_end:
                        pending = _Pending("entry", fill_time, boundary, new_side, regime)
                        forced_exit = signal.cycle_end + MINUTE
                elif side_btc and pending is None:
                    reverted = (side_btc == -1 and now <= 0) or (
                        side_btc == 1 and now >= 0
                    )
                    if reverted:
                        fill_time = boundary + HOUR + MINUTE
                        if forced_exit is None:
                            raise PairSimulationDataError("open pair has no terminal exit")
                        pending = _Pending("exit", min(fill_time, forced_exit),
                                           boundary, side_btc, regime)
    if expected_time != spec.end:
        raise PairSimulationDataError("minute series ended before fixed OOS end")
    if event_index != len(events):
        raise PairSimulationDataError("unprocessed funding events")
    if pending is not None or side_btc != 0 or forced_exit is not None:
        raise PairSimulationDataError("pair not flat at OOS end")
    for scenario in scenarios.values():
        scenario.finish_day()
        if len(scenario.daily_records) != (spec.end - spec.start).days:
            raise PairSimulationDataError("incomplete UTC daily path")
    expected_cycles = (spec.end - spec.start) // (spec.trading_hours * HOUR)
    if signal.completed_cycles != expected_cycles:
        raise PairSimulationDataError("wrong number of formed cycles")
    return {name: scenario.result(signal.completed_cycles)
            for name, scenario in scenarios.items()}


def run_pairs_simulation(bars: Iterable[object], funding_events: Iterable[object]
                         ) -> dict[str, SimulationResult]:
    """Run the sole frozen 2021–2025 experiment, including full input gates.

    Bars must start at 2020-01-01 00:00 UTC and end at 2025-12-31 23:59 UTC.
    Events must include each actual 2021–2025 BTC/ETH eight-hour funding time.
    The resulting paths are simulations, not executable bids or live orders.
    """
    return _simulate(bars, funding_events, spec=_FORMAL_SPEC)
