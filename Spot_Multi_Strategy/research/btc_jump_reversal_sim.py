"""One frozen, research-only BTC two-hour jump-reversal simulation.

Implements pre-registration 32bff50 and pre-result clarification 4bfac91.
There is no order-routing or parameter-search path.  Monthly data and funding
are supplied as validated iterators by the separate archive reader.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import random
from statistics import median
from typing import Iterable


MINUTE_MS = 60_000
TWO_HOURS_MS = 120 * MINUTE_MS
DAY_MS = 1_440 * MINUTE_MS
OOS_START_MS = 1_609_459_200_000
OOS_END_MS = 1_767_225_600_000
WARMUP_START_MS = 1_577_836_800_000
INITIAL_EQUITY = 10_000.0
FEE_RATE = 0.0005
ADVERSE_RATE = 0.0015
STRESS_ADVERSE_RATE = 0.0030


class DataGateError(ValueError):
    """The fixed data/causality gate failed; do not use partial PnL."""


class StrategyFailure(DataGateError):
    """The frozen strategy exhausted equity; seal it as a rejected candidate."""


@dataclass(frozen=True)
class MinuteBar:
    open_time_ms: int
    opening: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        if type(self.open_time_ms) is not int or self.open_time_ms % MINUTE_MS:
            raise DataGateError("minute bar must have an on-grid millisecond time")
        prices = (self.opening, self.high, self.low, self.close)
        if any(not math.isfinite(p) or p <= 0 for p in prices):
            raise DataGateError("minute bar has invalid price")
        if self.low > min(self.opening, self.close, self.high) or self.high < max(
            self.opening, self.close, self.low
        ):
            raise DataGateError("minute bar has inconsistent OHLC")


@dataclass(frozen=True)
class FundingRate:
    calc_time_ms: int
    rate: float

    def __post_init__(self) -> None:
        if type(self.calc_time_ms) is not int or self.calc_time_ms <= 0:
            raise DataGateError("funding timestamp must be positive milliseconds")
        if not math.isfinite(self.rate):
            raise DataGateError("funding rate must be finite")


class _RollingStd:
    def __init__(self, width: int = 360) -> None:
        self.width = width
        self.values: deque[float] = deque()
        self.total = 0.0
        self.squares = 0.0

    def sigma(self) -> float | None:
        if len(self.values) != self.width:
            return None
        variance = (self.squares - self.total * self.total / self.width) / (
            self.width - 1
        )
        return math.sqrt(max(0.0, variance))

    def append(self, value: float) -> None:
        if len(self.values) == self.width:
            old = self.values.popleft()
            self.total -= old
            self.squares -= old * old
        self.values.append(value)
        self.total += value
        self.squares += value * value


class TwoHourSignal:
    """Past-only 4-sigma signal state; return +1 long, -1 short, or 0."""

    def __init__(self) -> None:
        self.returns = _RollingStd()
        self.previous_close: float | None = None
        self.past_closes: deque[float] = deque(maxlen=1080)
        # Keep the current day's snapshot as well as 365 *prior* days.
        self.daily_volatility: deque[tuple[int, float]] = deque(maxlen=366)

    def observe(self, boundary_ms: int, close: float) -> tuple[int, str]:
        if boundary_ms % TWO_HOURS_MS or not math.isfinite(close) or close <= 0:
            raise DataGateError("invalid two-hour completed close")
        sigma = self.returns.sigma()  # Excludes the current return.
        past_90d = self.past_closes[0] if len(self.past_closes) == 1080 else None
        day_number = boundary_ms // DAY_MS
        past_daily_volatility = [
            value for day, value in self.daily_volatility if day < day_number
        ][-365:]
        regime = "unknown"
        if sigma is not None and len(past_daily_volatility) >= 300 and past_90d:
            trend = "bull" if close >= past_90d else "bear"
            vol = "high" if sigma >= median(past_daily_volatility) else "low"
            regime = trend + "_" + vol
        direction = 0
        if self.previous_close is not None:
            current_return = math.log(close / self.previous_close)
            if sigma is not None and sigma > 0 and abs(current_return) >= 4 * sigma:
                direction = -1 if current_return > 0 else 1
            self.returns.append(current_return)
        self.previous_close = close
        self.past_closes.append(close)
        # Midnight snapshot uses the same trailing, pre-current-return sigma.
        # It becomes available to regime classification on later days only.
        if boundary_ms % DAY_MS == 0 and sigma is not None:
            self.daily_volatility.append((day_number, sigma))
        return direction, regime


@dataclass
class _Position:
    side: int
    entry_ms: int
    exit_ms: int
    entry_fill: float
    quantity: float
    entry_equity: float
    regime: str
    funding_pnl: float = 0.0


@dataclass(frozen=True)
class Trade:
    side: str
    entry_ms: int
    exit_ms: int
    entry_fill: float
    exit_fill: float
    quantity: float
    net_pnl: float
    funding_pnl: float
    regime: str


@dataclass
class _Scenario:
    adverse_rate: float
    no_funding_credits: bool
    cash: float = INITIAL_EQUITY
    position: _Position | None = None
    peak: float = INITIAL_EQUITY
    max_drawdown: float = 0.0
    daily_equity: list[float] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    bankrupt: bool = False

    def enter(self, bar: MinuteBar, side: int, exit_ms: int, regime: str) -> None:
        if self.position is not None:
            raise DataGateError("overlapping position")
        if self.cash <= 0:
            self.bankrupt = True
            raise StrategyFailure("scenario equity became nonpositive before entry")
        entry_equity = self.cash
        fill = bar.opening * (1 + side * self.adverse_rate)
        quantity = entry_equity / fill
        self.cash -= FEE_RATE * quantity * fill
        self.position = _Position(
            side, bar.open_time_ms, exit_ms, fill, quantity, entry_equity, regime
        )

    def exit(self, bar: MinuteBar) -> None:
        position = self.position
        if position is None or bar.open_time_ms != position.exit_ms:
            raise DataGateError("scheduled exit is missing")
        fill = bar.opening * (1 - position.side * self.adverse_rate)
        self.cash += position.side * position.quantity * (fill - position.entry_fill)
        self.cash -= FEE_RATE * position.quantity * fill
        self.trades.append(Trade(
            "long" if position.side == 1 else "short",
            position.entry_ms, position.exit_ms, position.entry_fill, fill,
            position.quantity, self.cash - position.entry_equity,
            position.funding_pnl, position.regime,
        ))
        self.position = None
        if self.cash <= 0:
            self.bankrupt = True
            raise StrategyFailure("scenario equity became nonpositive at exit")

    def funding(self, rate: float, proxy_price: float) -> None:
        position = self.position
        if position is None:
            return
        pnl = -position.side * rate * position.quantity * proxy_price
        if self.no_funding_credits:
            pnl = min(0.0, pnl)
        self.cash += pnl
        position.funding_pnl += pnl

    def mark(self, bar: MinuteBar) -> float:
        position = self.position
        if position is None:
            close_equity = adverse_equity = self.cash
        else:
            close_equity = self.cash + position.side * position.quantity * (
                bar.close - position.entry_fill
            )
            adverse_price = bar.low if position.side == 1 else bar.high
            adverse_equity = self.cash + position.side * position.quantity * (
                adverse_price - position.entry_fill
            )
        if min(close_equity, adverse_equity) <= 0:
            raise StrategyFailure("marked scenario equity became nonpositive")
        # Both minute-end and adverse intraminute marks enter the drawdown.
        self.peak = max(self.peak, close_equity)
        if self.peak > 0:
            self.max_drawdown = max(
                self.max_drawdown, 1 - min(close_equity, adverse_equity) / self.peak
            )
        return close_equity


@dataclass(frozen=True)
class Simulation:
    baseline: _Scenario
    doubled_cost: _Scenario
    no_funding_credit: _Scenario
    funding_events_seen: int
    signal_count: int
    ignored_open_position_signals: int
    ignored_oos_tail_signals: int


def simulate(
    bars: Iterable[MinuteBar], funding: Iterable[FundingRate],
) -> Simulation:
    """Consume exactly 2020-2025 one-minute bars once, with no order output."""
    scenarios = (
        _Scenario(ADVERSE_RATE, False),
        _Scenario(STRESS_ADVERSE_RATE, False),
        _Scenario(ADVERSE_RATE, True),
    )
    signal = TwoHourSignal()
    funding_iter = iter(funding)
    next_funding = next(funding_iter, None)
    if next_funding is not None and not isinstance(next_funding, FundingRate):
        raise DataGateError("invalid funding event")
    prior_funding_ms = None
    expected_time = WARMUP_START_MS
    last_close: float | None = None
    pending: tuple[int, int, int, str] | None = None
    funding_count = signal_count = ignored_open = ignored_tail = 0
    for bar in bars:
        if not isinstance(bar, MinuteBar) or bar.open_time_ms != expected_time:
            raise DataGateError("missing, duplicate, or off-grid minute")
        expected_time += MINUTE_MS
        time_ms = bar.open_time_ms
        in_oos = OOS_START_MS <= time_ms < OOS_END_MS
        if in_oos:
            for scenario in scenarios:
                if scenario.position is not None and scenario.position.exit_ms == time_ms:
                    scenario.exit(bar)
            if pending is not None and pending[0] == time_ms:
                _, side, exit_ms, regime = pending
                for scenario in scenarios:
                    scenario.enter(bar, side, exit_ms, regime)
                pending = None
        while next_funding is not None and next_funding.calc_time_ms < time_ms + MINUTE_MS:
            if not isinstance(next_funding, FundingRate):
                raise DataGateError("invalid funding event")
            if next_funding.calc_time_ms < time_ms:
                raise DataGateError("funding event is missing an aligned price minute")
            if prior_funding_ms is not None and (
                next_funding.calc_time_ms <= prior_funding_ms
                or next_funding.calc_time_ms - prior_funding_ms > 12 * 60 * MINUTE_MS
            ):
                raise DataGateError("duplicate, unsorted, or gapped funding")
            if prior_funding_ms is None and (
                next_funding.calc_time_ms > OOS_START_MS + 12 * 60 * MINUTE_MS
            ):
                raise DataGateError("funding series begins too late")
            if last_close is None:
                raise DataGateError("funding proxy has no completed minute")
            for scenario in scenarios:
                scenario.funding(next_funding.rate, last_close)
            prior_funding_ms = next_funding.calc_time_ms
            funding_count += 1
            next_funding = next(funding_iter, None)
            if next_funding is not None and not isinstance(next_funding, FundingRate):
                raise DataGateError("invalid funding event")
        if in_oos:
            for scenario in scenarios:
                equity = scenario.mark(bar)
                if (time_ms + MINUTE_MS) % DAY_MS == 0:
                    scenario.daily_equity.append(equity)
        last_close = bar.close
        boundary_ms = time_ms + MINUTE_MS
        if boundary_ms % TWO_HOURS_MS == 0:
            direction, regime = signal.observe(boundary_ms, bar.close)
            if OOS_START_MS <= boundary_ms < OOS_END_MS and direction:
                signal_count += 1
                exit_ms = boundary_ms + 121 * MINUTE_MS
                if exit_ms >= OOS_END_MS:
                    ignored_tail += 1
                elif scenarios[0].position is not None or pending is not None:
                    ignored_open += 1
                else:
                    pending = (boundary_ms + MINUTE_MS, direction, exit_ms, regime)
    if expected_time != OOS_END_MS:
        raise DataGateError("2020-2025 data ends before complete fixed grid")
    if next_funding is not None and next_funding.calc_time_ms < OOS_END_MS:
        raise DataGateError("unconsumed funding inside OOS")
    if prior_funding_ms is None or OOS_END_MS - prior_funding_ms > 12 * 60 * MINUTE_MS:
        raise DataGateError("funding series ends too early")
    if pending is not None or any(s.position is not None for s in scenarios):
        raise DataGateError("position or entry extends beyond OOS")
    if any(len(s.daily_equity) != 1826 for s in scenarios):
        raise DataGateError("OOS daily equity does not have 1,826 days")
    if any(s.bankrupt for s in scenarios):
        raise StrategyFailure("scenario equity became nonpositive")
    return Simulation(*scenarios, funding_count, signal_count, ignored_open, ignored_tail)


def _daily_returns(daily_equity: list[float], *, start: float) -> list[float]:
    previous = start
    result = []
    for equity in daily_equity:
        if previous <= 0 or not math.isfinite(equity):
            raise DataGateError("nonpositive or nonfinite marked equity")
        result.append(equity / previous - 1)
        previous = equity
    return result


def _path_metrics(returns: list[float], initial: float = INITIAL_EQUITY) -> dict:
    """Recompound a daily net-return path, including dollar-PnL PF."""
    equity = peak = initial
    positive = negative = 0.0
    max_dd = 0.0
    for ret in returns:
        if not math.isfinite(ret) or ret <= -1:
            return {"cagr": -1.0, "sharpe": None, "max_drawdown": 1.0, "profit_factor": None}
        pnl = equity * ret
        if pnl > 0:
            positive += pnl
        elif pnl < 0:
            negative -= pnl
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)
    n = len(returns)
    mean = sum(returns) / n if n else 0
    variance = sum((ret - mean) ** 2 for ret in returns) / (n - 1) if n > 1 else 0
    sharpe = math.sqrt(365) * mean / math.sqrt(variance) if variance > 0 else None
    return {
        "cagr": (equity / initial) ** (365.25 / n) - 1 if n and equity > 0 else None,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": positive / negative if positive > 0 and negative > 0 else None,
    }


def _passes(metrics: dict, *, drawdown: float | None = None) -> bool:
    cagr, sharpe, pf = (
        metrics["cagr"], metrics["sharpe"], metrics["profit_factor"]
    )
    return (
        cagr is not None and cagr >= 0.40
        and sharpe is not None and sharpe > 1.5
        and (metrics["max_drawdown"] if drawdown is None else drawdown) <= 0.20
        and pf is not None and pf > 1.3
    )


def monte_carlo(
    daily_returns: list[float], *, paths: int = 5_000, seed: int = 20261107,
    block_days: int = 21,
) -> dict:
    if not daily_returns or paths <= 0 or block_days <= 0:
        raise ValueError("Monte Carlo requires a nonempty path and positive settings")
    rng = random.Random(seed)
    size = len(daily_returns)
    scores = {"cagr": [], "sharpe": [], "max_drawdown": [], "profit_factor": []}
    joint = 0
    for _ in range(paths):
        sample = []
        while len(sample) < size:
            start = rng.randrange(size)
            sample.extend(
                daily_returns[(start + offset) % size]
                for offset in range(min(block_days, size - len(sample)))
            )
        metrics = _path_metrics(sample)
        joint += int(_passes(metrics))
        for key in scores:
            if metrics[key] is not None:
                scores[key].append(metrics[key])

    def percentiles(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"p05": None, "p50": None, "p95": None}
        values.sort()
        index = lambda p: min(len(values) - 1, round((len(values) - 1) * p))
        return {"p05": values[index(0.05)], "p50": values[index(0.5)],
                "p95": values[index(0.95)]}

    return {
        "seed": seed, "block_days": block_days, "paths": paths,
        "joint_pass_count": joint, "joint_pass_fraction": joint / paths,
        "distributions": {
            key: {**percentiles(value), "undefined_count": paths - len(value)}
            for key, value in scores.items()
        },
    }


def report(simulation: Simulation) -> dict:
    """Compute fixed Gate diagnostics from a complete OOS simulation."""
    baseline = simulation.baseline
    returns = _daily_returns(baseline.daily_equity, start=INITIAL_EQUITY)
    metrics = _path_metrics(returns)
    metrics["max_drawdown"] = max(metrics["max_drawdown"], baseline.max_drawdown)
    years = {}
    start = INITIAL_EQUITY
    offset = 0
    for year in range(2021, 2026):
        days = 366 if year == 2024 else 365
        segment = baseline.daily_equity[offset:offset + days]
        if len(segment) != days:
            raise DataGateError("annual fold is incomplete")
        fold_returns = _daily_returns(segment, start=start)
        fold_metrics = _path_metrics(fold_returns, initial=start)
        trade_count = sum(
            datetime.fromtimestamp(trade.entry_ms / 1000, timezone.utc).year == year
            for trade in baseline.trades
        )
        years[str(year)] = {
            **fold_metrics,
            "net_return": segment[-1] / start - 1,
            "trades_entered": trade_count,
            "passes": segment[-1] > start and (
                fold_metrics["sharpe"] is not None and fold_metrics["sharpe"] > 0
            ) and trade_count > 0,
        }
        start = segment[-1]
        offset += days

    months = []
    previous = INITIAL_EQUITY
    day_index = 0
    for year in range(2021, 2026):
        for month in range(1, 13):
            if month == 12:
                next_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc)
            month_end_index = (int(next_month.timestamp() * 1000) - OOS_START_MS) // DAY_MS - 1
            ending = baseline.daily_equity[month_end_index]
            months.append(ending / previous - 1)
            previous = ending
            day_index = month_end_index + 1
    if day_index != 1826:
        raise DataGateError("monthly accounting does not cover OOS")
    streak = worst_streak = 0
    for value in months:
        streak = streak + 1 if value < 0 else 0
        worst_streak = max(worst_streak, streak)
    monthly = {
        "positive": sum(value > 0 for value in months),
        "negative": sum(value < 0 for value in months),
        "flat": sum(value == 0 for value in months),
        "longest_losing_streak": worst_streak,
        "worst_return": min(months),
    }
    regime = {
        key: {"trades": 0, "net_pnl": 0.0}
        for key in ("bull_high", "bull_low", "bear_high", "bear_low", "unknown")
    }
    for trade in baseline.trades:
        row = regime[trade.regime]
        row["trades"] += 1
        row["net_pnl"] += trade.net_pnl
    regime_pass = all(
        regime[key]["trades"] > 0 and regime[key]["net_pnl"] > 0
        for key in ("bull_high", "bull_low", "bear_high", "bear_low")
    ) and regime["unknown"]["trades"] == 0
    mc = monte_carlo(returns)
    cost_stress_equity = simulation.doubled_cost.daily_equity[-1]
    no_funding_equity = simulation.no_funding_credit.daily_equity[-1]
    gate = {
        "cost_net_metrics": _passes(metrics),
        "annual_walk_forward": all(row["passes"] for row in years.values()),
        "regime_robustness": regime_pass,
        "monte_carlo": mc["joint_pass_fraction"] >= 0.80,
        "doubled_adverse_cost_profitable": cost_stress_equity > INITIAL_EQUITY,
        "independent_executable_evidence": False,
    }
    return {
        "experiment_id": "btc_2h_jump_reversal_2026_11_07",
        "preregistration_commit": "32bff50",
        "amendment_commit": "4bfac91",
        "evidence_grade": "retrospective_blind_historical_holdout",
        "metrics": metrics,
        "monthly": monthly,
        "annual_folds": years,
        "regimes": regime,
        "monte_carlo": mc,
        "trade_count": len(baseline.trades),
        "funding_events_seen": simulation.funding_events_seen,
        "signal_count": simulation.signal_count,
        "ignored_open_position_signals": simulation.ignored_open_position_signals,
        "ignored_oos_tail_signals": simulation.ignored_oos_tail_signals,
        "doubled_cost_ending_equity": cost_stress_equity,
        "no_funding_credit_ending_equity": no_funding_equity,
        "gate": gate,
        "full_gate_pass": all(gate.values()),
    }
