"""Pure, frozen simulation for the network-adoption preregistration.

This module performs no I/O and never fetches or selects historical data.  Inputs
are canonical in-memory maps supplied by independently checked archive loaders.
The strategy/evaluation constants follow the separately committed preregistration.
Do not tune this code against the 2021--2025 outcome.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
import random
from statistics import median
from typing import Mapping, Sequence


UTC = timezone.utc
SYMBOLS = (
    "BATUSDT", "CVCUSDT", "KNCUSDT", "MKRUSDT", "OMGUSDT", "RENUSDT", "ZRXUSDT",
)
FEE_RATE = 0.0005
MIN_DAILY_QUOTE_VOLUME = 5_000_000.0
INITIAL_EQUITY = 10_000.0


class DataIneligibleError(ValueError):
    """The preregistered data/causality checks cannot be met."""


@dataclass(frozen=True)
class StressPoint:
    # period_date assigns the terminal 2026-01-01 exit to 2025-12-31.
    period_date: date
    open_time: datetime
    open_equity: float
    stress_equity: float


@dataclass(frozen=True)
class SimulationResult:
    multiplier: int
    slip_bps: float
    start: datetime
    end: datetime
    daily_returns: tuple[tuple[date, float], ...]
    daily_equity: tuple[tuple[date, float], ...]
    stress_points: tuple[StressPoint, ...]
    stress_drawdown: float
    daily_drawdown: float
    fee_usdt: float
    slippage_usdt: float
    funding_usdt: float
    turnover_usdt: float
    trade_count: int
    capacity_failed: bool
    leverage_failed: bool
    liquidated: bool
    cash_days: int
    active_days: int
    exposure_days: tuple[date, ...]
    # The source lacks verified historical as-of vintages, irrespective of PnL.
    independent_oos: bool = False
    evidence_grade: str = "retrospective_source_without_verified_vintages"


@dataclass(frozen=True)
class ReturnMetrics:
    observations: int
    cagr: float
    sharpe: float | None
    max_drawdown: float
    profit_factor: float
    ending_multiple: float
    negative_days: int


@dataclass(frozen=True)
class MonthStability:
    months: int
    positive_months: int
    positive_share: float
    worst_month: float
    longest_losing_streak: int
    returns: tuple[tuple[str, float], ...]
    passed: bool


@dataclass(frozen=True)
class FoldResult:
    year: int
    return_net: float
    profit_factor: float
    stress_drawdown: float
    passed: bool


@dataclass(frozen=True)
class RegimeResult:
    name: str
    days: int
    active_days: int
    cash_days: int
    compounded_return: float
    sharpe: float | None
    max_drawdown: float
    profit_factor: float


@dataclass(frozen=True)
class MonteCarloResult:
    paths: int
    block_days: int
    seed: int
    p05_cagr: float
    p05_sharpe: float | None
    p05_daily_drawdown: float
    p05_profit_factor: float
    joint_pass_share: float
    passed: bool


@dataclass(frozen=True)
class Evaluation:
    metrics: ReturnMetrics
    stress_drawdown: float
    months: MonthStability
    folds: tuple[FoldResult, ...]
    regimes: tuple[RegimeResult, ...]
    monte_carlo: MonteCarloResult
    numerical_gate: bool
    full_gate: bool


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise DataIneligibleError("naive UTC timestamp")
    return value.astimezone(UTC)


def _finite_positive(value: object, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DataIneligibleError(f"invalid {label}") from exc
    if not math.isfinite(number) or number <= 0:
        raise DataIneligibleError(f"invalid {label}")
    return number


def _bar_open(bar: object) -> float:
    return _finite_positive(getattr(bar, "open", None), "bar open")


def _mark1_open(value: object) -> float:
    return _finite_positive(getattr(value, "open", value), "1h mark open")


def address_growth_score(
    points: Mapping[date, tuple[float, datetime]], decision_time: datetime,
) -> float | None:
    """35 contiguous positive days through prior Friday; no fill or revised-vintage claim."""
    decision_time = _utc(decision_time)
    if decision_time.weekday() != 0 or decision_time.hour != 8 or decision_time.minute != 0:
        raise ValueError("score decision must be Monday 08:00 UTC")
    friday = decision_time.date() - timedelta(days=3)
    recent: list[float] = []
    prior: list[float] = []
    for offset in range(34, -1, -1):
        item = points.get(friday - timedelta(days=offset))
        if item is None:
            return None
        count, completed = item
        try:
            value = float(count)
            completed = _utc(completed)
        except (TypeError, ValueError, DataIneligibleError):
            return None
        if not math.isfinite(value) or value <= 0 or completed > decision_time:
            return None
        (prior if offset >= 7 else recent).append(value)
    if len(prior) != 28 or len(recent) != 7:
        return None
    return math.log((sum(recent) / 7.0) / (sum(prior) / 28.0))


def _daily_volume(
    trade: Mapping[datetime, object], day: date,
) -> float | None:
    total = 0.0
    for hour in (0, 4, 8, 12, 16, 20):
        stamp = datetime.combine(day, time(hour=hour), UTC)
        bar = trade.get(stamp)
        if bar is None:
            return None
        volume = getattr(bar, "quote_volume", None)
        try:
            volume = float(volume)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(volume) or volume < 0:
            return None
        total += volume
    return total


def _last30_volumes(
    trade: Mapping[datetime, object], asof_day: date,
) -> list[float] | None:
    # asof_day is the last fully completed UTC day, never the execution day.
    values = []
    for offset in range(29, -1, -1):
        amount = _daily_volume(trade, asof_day - timedelta(days=offset))
        if amount is None:
            return None
        values.append(amount)
    return values


def _is_valid_4h_bar(bar: object, *, volume_required: bool) -> bool:
    try:
        op = _bar_open(bar)
        high = _finite_positive(getattr(bar, "high", None), "bar high")
        low = _finite_positive(getattr(bar, "low", None), "bar low")
        close = _finite_positive(getattr(bar, "close", None), "bar close")
        if low > min(op, close) or high < max(op, close) or high < low:
            return False
        if volume_required:
            volume = float(getattr(bar, "quote_volume", None))
            if not math.isfinite(volume) or volume < 0:
                return False
    except (DataIneligibleError, TypeError, ValueError):
        return False
    return True


def run_backtest(
    adrs: Mapping[str, Mapping[date, tuple[float, datetime]]],
    trade4: Mapping[str, Mapping[datetime, object]],
    mark4: Mapping[str, Mapping[datetime, object]],
    mark1: Mapping[str, Mapping[datetime, object]],
    funding: Mapping[str, Mapping[datetime, float]],
    *,
    announcements: Mapping[str, datetime] | None = None,
    multiplier: int = 1,
    slip_bps: float = 15.0,
    start: datetime = datetime(2021, 1, 1, tzinfo=UTC),
    end: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    symbols: Sequence[str] = SYMBOLS,
    initial_equity: float = INITIAL_EQUITY,
    strict_universe_bars: bool = True,
) -> SimulationResult:
    """Run one frozen multiplier scenario, including funding, costs and stress.

    The caller must run 1x first.  Calling 3x/5x is permitted only after 1x
    net Sharpe >1.5 and daily PF >1.3; enforce via ``run_registered_scenarios``.
    Custom date/symbol arguments exist solely for synthetic unit tests.
    """
    start, end = _utc(start), _utc(end)
    if (start >= end or start.time() != time(0) or end.time() != time(0)
            or (end - start).seconds != 0):
        raise ValueError("start/end must be midnight UTC and span whole days")
    if multiplier not in (1, 3, 5) or not math.isfinite(slip_bps) or slip_bps < 0:
        raise ValueError("unregistered leverage/slippage")
    initial_equity = _finite_positive(initial_equity, "initial equity")
    symbols = tuple(symbols)
    if len(symbols) != len(set(symbols)) or len(symbols) < 4:
        raise ValueError("need at least four unique registered symbols")
    notices = {sym: _utc(value) for sym, value in (announcements or {}).items() if value is not None}
    for sym in symbols:
        if sym not in adrs or sym not in trade4 or sym not in mark4 or sym not in mark1 or sym not in funding:
            raise DataIneligibleError(f"missing canonical map for {sym}")
    # Amendment 01 (dd77ed7): preserve the raw timestamp; 0--100ms after a
    # whole UTC hour maps to that logical hour for *ordering* and mark lookup.
    # Wider offsets and canonical duplicates are fatal only when active.
    funding_by_bar: dict[datetime, list[tuple[datetime, datetime, str, float]]] = defaultdict(list)
    for sym in symbols:
        for event_time, rate in funding[sym].items():
            event = _utc(event_time)
            try:
                rate = float(rate)
            except (TypeError, ValueError) as exc:
                raise DataIneligibleError(f"invalid funding rate {sym} {event}") from exc
            if not math.isfinite(rate):
                raise DataIneligibleError(f"invalid funding rate {sym} {event}")
            logical_hour = event.replace(minute=0, second=0, microsecond=0)
            if start <= logical_hour <= end:
                bar_time = logical_hour.replace(hour=(logical_hour.hour // 4) * 4)
                funding_by_bar[bar_time].append((logical_hour, event, sym, rate))
    for events in funding_by_bar.values():
        events.sort(key=lambda item: (item[0], item[2], item[1]))

    cash = initial_equity
    positions = {sym: 0.0 for sym in symbols}
    fee_usdt = slippage_usdt = funding_usdt = turnover_usdt = 0.0
    trades = 0
    capacity_failed = leverage_failed = liquidated = False
    pending: tuple[datetime, tuple[str, ...], tuple[str, ...]] | None = None
    risk_reduce_due: datetime | None = None
    resume_at = start
    daily_ret: list[tuple[date, float]] = []
    daily_eq: list[tuple[date, float]] = []
    stress_points: list[StressPoint] = []
    active_dates: set[date] = set()
    last_midnight_equity = initial_equity
    peak_daily = initial_equity
    peak_stress = initial_equity
    daily_mdd = stress_mdd = 0.0
    volume_cache: dict[tuple[str, date], list[float] | None] = {}

    def volumes(sym: str, asof_day: date) -> list[float] | None:
        key = (sym, asof_day)
        if key not in volume_cache:
            volume_cache[key] = _last30_volumes(trade4[sym], asof_day)
        return volume_cache[key]

    def mark_equity(stamp: datetime) -> float:
        equity = cash
        for sym, q in positions.items():
            if q:
                bar = mark4[sym].get(stamp)
                if bar is None or not _is_valid_4h_bar(bar, volume_required=False):
                    raise DataIneligibleError(f"missing/invalid active mark4 {sym} {stamp}")
                equity += q * _bar_open(bar)
        return equity

    def execute(targets: Mapping[str, float], stamp: datetime) -> None:
        nonlocal cash, fee_usdt, slippage_usdt, turnover_usdt, trades, capacity_failed
        # Targets were calculated from one common pre-trade equity.  Price and
        # fee effects do not feed back into the other legs' target quantities.
        for sym in symbols:
            delta = targets.get(sym, 0.0) - positions[sym]
            if abs(delta) < 1e-14:
                continue
            bar = trade4[sym].get(stamp)
            if bar is None or not _is_valid_4h_bar(bar, volume_required=True):
                raise DataIneligibleError(f"missing/invalid executable trade4 {sym} {stamp}")
            op = _bar_open(bar)
            before = abs(delta * op)
            thirty = volumes(sym, stamp.date() - timedelta(days=1))
            if thirty is None:
                raise DataIneligibleError(f"missing 30d execution volume {sym} {stamp}")
            if before > 0.01 * median(thirty) + 1e-9:
                capacity_failed = True
            slip = slip_bps / 10_000.0
            fill = op * (1.0 + slip if delta > 0 else 1.0 - slip)
            fee = FEE_RATE * abs(delta * fill)
            cash -= delta * fill + fee
            positions[sym] = targets.get(sym, 0.0)
            fee_usdt += fee
            slippage_usdt += before * slip
            turnover_usdt += before
            trades += 1

    def apply_funding_event(logical: datetime, raw: datetime, sym: str, rate: float) -> None:
        nonlocal cash, funding_usdt
        q = positions[sym]
        if not q:
            return
        offset_ms = (raw - logical).total_seconds() * 1000.0
        if offset_ms < 0 or offset_ms > 100:
            raise DataIneligibleError(f"active funding offset >100ms {sym} {raw}")
        mark = mark1[sym].get(logical)
        if mark is None:
            raise DataIneligibleError(f"missing settlement mark1 {sym} {logical}")
        payment = -q * _mark1_open(mark) * rate
        cash += payment
        funding_usdt += payment

    def settle_funding(stamp: datetime) -> None:
        seen: set[str] = set()
        for logical, raw, sym, rate in funding_by_bar.get(
            stamp.replace(hour=(stamp.hour // 4) * 4, minute=0, second=0, microsecond=0), ()
        ):
            if logical != stamp:
                continue
            if not positions[sym]:
                continue
            if sym in seen:
                raise DataIneligibleError(f"duplicate canonical funding {sym} {logical}")
            seen.add(sym)
            apply_funding_event(logical, raw, sym, rate)

    def settle_intrabar(bar_time: datetime) -> float:
        nonlocal cash, funding_usdt
        minimum = cash
        seen: set[tuple[datetime, str]] = set()
        for logical, raw, sym, rate in funding_by_bar.get(bar_time, ()):
            if logical <= bar_time:
                continue
            if not positions[sym]:
                continue
            key = (logical, sym)
            if key in seen:
                raise DataIneligibleError(f"duplicate canonical funding {sym} {logical}")
            seen.add(key)
            apply_funding_event(logical, raw, sym, rate)
            minimum = min(minimum, cash)
        return minimum

    def selected_at(decision: datetime) -> tuple[tuple[str, ...], tuple[str, ...]]:
        candidates: list[tuple[float, str]] = []
        asof_day = decision.date() - timedelta(days=1)
        for sym in symbols:
            if sym in notices and notices[sym] < decision:
                continue
            score = address_growth_score(adrs[sym], decision)
            vols = volumes(sym, asof_day)
            if score is None or vols is None or sum(vols) / 30.0 < MIN_DAILY_QUOTE_VOLUME:
                continue
            candidates.append((score, sym))
        if len(candidates) < 4:
            return (), ()
        # descending score, ascending symbol for ties; bottom two are the
        # final two of this same deterministic ordering.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return tuple(sym for _, sym in candidates[:2]), tuple(sym for _, sym in candidates[-2:])

    stamp = start
    while stamp <= end:
        settle_funding(stamp)  # old positions own the settlement at this instant
        if stamp == end:
            if any(positions.values()):
                execute({sym: 0.0 for sym in symbols}, stamp)
            if cash <= 0:
                liquidated = True
                cash = 0.0
            ending = cash
            stress_points.append(StressPoint((end - timedelta(days=1)).date(), stamp, ending, ending))
            peak_stress = max(peak_stress, ending)
            stress_mdd = min(stress_mdd, ending / peak_stress - 1.0)
            previous_day = (stamp - timedelta(days=1)).date()
            ret = ending / last_midnight_equity - 1.0 if last_midnight_equity > 0 else 0.0
            daily_ret.append((previous_day, max(-1.0, ret)))
            daily_eq.append((previous_day, ending))
            peak_daily = max(peak_daily, ending)
            daily_mdd = min(daily_mdd, ending / peak_daily - 1.0)
            break

        # An unannounced active market cannot silently disappear.  Already
        # announced markets remain in the fixed pool but are no longer eligible.
        if strict_universe_bars:
            for sym in symbols:
                if sym in notices and notices[sym] < stamp and positions[sym] == 0:
                    continue
                trade_bar = trade4[sym].get(stamp)
                mark_bar = mark4[sym].get(stamp)
                if not _is_valid_4h_bar(trade_bar, volume_required=True) or not _is_valid_4h_bar(mark_bar, volume_required=False):
                    raise DataIneligibleError(f"unexplained 4h archive gap {sym} {stamp}")

        # The first opening strictly after public notice closes *the entire*
        # portfolio if the announced leg is held; no replacement until next
        # normal Monday 12:00.  This dominates a same-open leverage cut, so
        # the exit is never charged twice.  A non-held notice only bars entry.
        held_notice = any(positions[sym] and sym in notices and notices[sym] < stamp for sym in symbols)
        if held_notice and not liquidated:
            execute({sym: 0.0 for sym in symbols}, stamp)
            pending = None
            risk_reduce_due = None
            days_to_monday = (7 - stamp.weekday()) % 7
            next_monday = datetime.combine(stamp.date() + timedelta(days=days_to_monday), time(12), UTC)
            if next_monday <= stamp:
                next_monday += timedelta(days=7)
            resume_at = next_monday

        # Funding has already been paid.  Any previously observed 5x breach
        # that was not superseded by full notice exit is reduced at this next
        # executable open; the earlier breach remains flagged.
        if risk_reduce_due == stamp and any(positions.values()) and not liquidated:
            equity = mark_equity(stamp)
            gross = sum(abs(q * _bar_open(mark4[sym][stamp])) for sym, q in positions.items())
            if equity <= 0:
                liquidated = True
                cash = 0.0
                positions = {sym: 0.0 for sym in symbols}
            elif gross > 5.0 * equity:
                scale = max(0.0, 5.0 * equity / gross)
                execute({sym: positions[sym] * scale for sym in symbols}, stamp)
            risk_reduce_due = None

        if stamp.weekday() == 0 and stamp.hour == 8 and not liquidated:
            longs, shorts = selected_at(stamp)
            pending = (stamp, longs, shorts)

        if stamp.weekday() == 0 and stamp.hour == 12 and pending is not None and not liquidated:
            decision, longs, shorts = pending
            # A notice published in the 08:00--12:00 delay cancels the entire
            # pending rebalance if it concerns a planned target leg.
            canceled = any(
                sym in notices and decision <= notices[sym] < stamp
                for sym in longs + shorts
            )
            if not canceled and stamp >= resume_at:
                equity = mark_equity(stamp)
                if equity <= 0:
                    liquidated = True
                    cash = 0.0
                    positions = {sym: 0.0 for sym in symbols}
                else:
                    target = {sym: 0.0 for sym in symbols}
                    for sym in longs:
                        target[sym] = +0.25 * multiplier * equity / _bar_open(trade4[sym][stamp])
                    for sym in shorts:
                        target[sym] = -0.25 * multiplier * equity / _bar_open(trade4[sym][stamp])
                    execute(target, stamp)
            pending = None

        equity_open = mark_equity(stamp) if not liquidated else cash
        if equity_open <= 0:
            liquidated = True
            cash = 0.0
            positions = {sym: 0.0 for sym in symbols}
            equity_open = 0.0
        peak_stress = max(peak_stress, equity_open)
        gross_open = sum(
            abs(q * _bar_open(mark4[sym][stamp])) for sym, q in positions.items() if q
        )
        if gross_open > 5.0 * equity_open + 1e-9:
            leverage_failed = True
            risk_reduce_due = stamp + timedelta(hours=4)

        # A UTC day ends at this exact open, after its same-time funding and
        # orders.  Intrabar t+1/2/3h settlements and adverse marks belong to
        # the *new* day and must never leak into the preceding day's NAV.
        if stamp.hour == 0 and stamp != start:
            previous_day = (stamp - timedelta(days=1)).date()
            ret = equity_open / last_midnight_equity - 1.0 if last_midnight_equity > 0 else 0.0
            daily_ret.append((previous_day, max(-1.0, ret)))
            daily_eq.append((previous_day, equity_open))
            peak_daily = max(peak_daily, equity_open)
            daily_mdd = min(daily_mdd, equity_open / peak_daily - 1.0)
            last_midnight_equity = equity_open
        elif stamp == start:
            last_midnight_equity = equity_open

        if positions and any(positions.values()):
            active_dates.add(stamp.date())
        min_cash = settle_intrabar(stamp)
        stress_equity = min_cash
        stress_gross = 0.0
        if not liquidated:
            for sym, q in positions.items():
                if not q:
                    continue
                bar = mark4[sym].get(stamp)
                if bar is None or not _is_valid_4h_bar(bar, volume_required=False):
                    raise DataIneligibleError(f"missing/invalid stress mark4 {sym} {stamp}")
                adverse = float(bar.low if q > 0 else bar.high)
                stress_equity += q * adverse
                stress_gross += abs(q * adverse)
            stress_mdd = min(stress_mdd, max(0.0, stress_equity) / peak_stress - 1.0)
            if stress_gross > 0 and stress_equity <= 0.01 * stress_gross:
                # Conservative intrabar simultaneous adverse extrema plus the
                # lowest cash around intrabar settlements.  It is a stress
                # estimate, not a claim of a reproducible simultaneous tick.
                liquidation_cash = min_cash
                for sym, q in positions.items():
                    if not q:
                        continue
                    bar = mark4[sym][stamp]
                    adverse = float(bar.low if q > 0 else bar.high)
                    delta = -q
                    fill = adverse * (1 + slip_bps / 10_000.0 if delta > 0 else 1 - slip_bps / 10_000.0)
                    fee = FEE_RATE * abs(delta * fill)
                    liquidation_cash -= delta * fill + fee
                    fee_usdt += fee
                    slippage_usdt += abs(delta * adverse) * slip_bps / 10_000.0
                    turnover_usdt += abs(delta * adverse)
                    trades += 1
                liquidation_cash -= 0.01 * stress_gross
                fee_usdt += 0.01 * stress_gross
                cash = max(0.0, liquidation_cash)
                positions = {sym: 0.0 for sym in symbols}
                liquidated = True
                stress_equity = max(0.0, min(stress_equity, cash))
                stress_mdd = min(stress_mdd, stress_equity / peak_stress - 1.0)
        stress_points.append(StressPoint(stamp.date(), stamp, equity_open, stress_equity))
        stamp += timedelta(hours=4)

    if len(daily_ret) != (end - start).days:
        raise DataIneligibleError("daily return calendar is incomplete")
    active_count = sum(day in active_dates for day, _ in daily_ret)
    return SimulationResult(
        multiplier=multiplier, slip_bps=slip_bps, start=start, end=end,
        daily_returns=tuple(daily_ret), daily_equity=tuple(daily_eq),
        stress_points=tuple(stress_points), stress_drawdown=stress_mdd,
        daily_drawdown=daily_mdd, fee_usdt=fee_usdt,
        slippage_usdt=slippage_usdt, funding_usdt=funding_usdt,
        turnover_usdt=turnover_usdt, trade_count=trades,
        capacity_failed=capacity_failed, leverage_failed=leverage_failed,
        liquidated=liquidated, cash_days=len(daily_ret) - active_count,
        active_days=active_count, exposure_days=tuple(sorted(active_dates)),
    )


def _returns_metrics(values: Sequence[float], *, start_equity: float = 1.0) -> ReturnMetrics:
    n = len(values)
    if not n:
        raise ValueError("empty returns")
    equity = start_equity
    peak = equity
    mdd = 0.0
    mean = m2 = positive = negative = 0.0
    count = 0
    negative_days = 0
    for ret in values:
        if not math.isfinite(ret) or ret < -1.0:
            raise DataIneligibleError("invalid daily return")
        equity *= 1.0 + ret
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
        count += 1
        delta = ret - mean
        mean += delta / count
        m2 += delta * (ret - mean)
        positive += max(ret, 0.0)
        negative += min(ret, 0.0)
        if ret < 0:
            negative_days += 1
    # Welford's sample variance (ddof=1) avoids cancellation for tiny but
    # genuinely nonzero variation.  The preregistration rejects only zero or
    # missing standard deviation, not an arbitrary low-variance threshold.
    variance = max(0.0, m2 / (n - 1)) if n > 1 else 0.0
    sharpe = math.sqrt(365.25) * mean / math.sqrt(variance) if variance > 0 else None
    cagr = (equity / start_equity) ** (365.25 / n) - 1.0 if equity > 0 else -1.0
    pf = positive / -negative if negative < 0 else math.inf
    return ReturnMetrics(n, cagr, sharpe, mdd, pf, equity / start_equity, negative_days)


def summarize_returns(result: SimulationResult) -> ReturnMetrics:
    return _returns_metrics([ret for _, ret in result.daily_returns])


def month_stability(result: SimulationResult) -> MonthStability:
    months: dict[str, float] = {}
    for day, ret in result.daily_returns:
        key = day.strftime("%Y-%m")
        months[key] = months.get(key, 1.0) * (1.0 + ret)
    monthly = tuple((month, value - 1.0) for month, value in sorted(months.items()))
    positive_count = sum(value > 0 for _, value in monthly)
    worst = min((value for _, value in monthly), default=0.0)
    streak = longest = 0
    for _, value in monthly:
        streak = streak + 1 if value < 0 else 0
        longest = max(longest, streak)
    share = positive_count / len(monthly) if monthly else 0.0
    return MonthStability(len(monthly), positive_count, share, worst, longest, monthly,
                          bool(monthly) and share >= 0.70 and worst >= -0.10)


def yearly_folds(result: SimulationResult) -> tuple[FoldResult, ...]:
    by_year: dict[int, list[float]] = defaultdict(list)
    for day, ret in result.daily_returns:
        by_year[day.year].append(ret)
    equity_by_day = dict(result.daily_equity)
    folds = []
    for year, values in sorted(by_year.items()):
        if len(values) != (date(year + 1, 1, 1) - date(year, 1, 1)).days:
            raise DataIneligibleError(f"incomplete annual fold {year}")
        metrics = _returns_metrics(values)
        if year == result.start.year:
            beginning = INITIAL_EQUITY
        else:
            beginning = equity_by_day[date(year - 1, 12, 31)]
        peak = beginning
        mdd = 0.0
        if beginning > 0:
            for point in result.stress_points:
                if point.period_date.year != year:
                    continue
                peak = max(peak, point.open_equity)
                mdd = min(mdd, point.stress_equity / peak - 1.0)
            # The Dec 31 return ends at Jan 1 00:00.  Its same-time funding
            # and any scheduled exit belong to the old annual fold; the
            # following Jan 1 intrabar stress does not.
            boundary_equity = equity_by_day[date(year, 12, 31)]
            peak = max(peak, boundary_equity)
            mdd = min(mdd, boundary_equity / peak - 1.0)
        folds.append(FoldResult(year, metrics.ending_multiple - 1.0,
                                metrics.profit_factor, mdd,
                                beginning > 0 and metrics.ending_multiple > 1.0
                                and metrics.profit_factor > 1.0 and mdd >= -0.20))
    return tuple(folds)


def classify_btc_regimes(
    result: SimulationResult, btc_daily_close: Mapping[date, float],
) -> dict[date, str]:
    labels = {}
    for day, _ in result.daily_returns:
        yesterday = day - timedelta(days=1)
        before = day - timedelta(days=31)
        if yesterday not in btc_daily_close or before not in btc_daily_close:
            raise DataIneligibleError(f"missing BTC regime close {day}")
        change = _finite_positive(btc_daily_close[yesterday], "BTC close") / _finite_positive(btc_daily_close[before], "BTC close") - 1.0
        labels[day] = "bull" if change > 0.10 else "bear" if change < -0.10 else "sideways"
    return labels


def regime_results(
    result: SimulationResult, btc_daily_close: Mapping[date, float],
) -> tuple[RegimeResult, ...]:
    labels = classify_btc_regimes(result, btc_daily_close)
    active_dates = set(result.exposure_days)
    by_regime: dict[str, list[float]] = {name: [] for name in ("bull", "bear", "sideways")}
    for day, ret in result.daily_returns:
        by_regime[labels[day]].append(ret)
    out = []
    for name in ("bull", "bear", "sideways"):
        values = by_regime[name]
        if not values:
            out.append(RegimeResult(name, 0, 0, 0, 0.0, None, 0.0, math.inf))
            continue
        metrics = _returns_metrics(values)
        active = sum(day in active_dates for day, _ in result.daily_returns if labels[day] == name)
        out.append(RegimeResult(name, len(values), active, len(values) - active,
                                metrics.ending_multiple - 1.0, metrics.sharpe,
                                metrics.max_drawdown, metrics.profit_factor))
    return tuple(out)


def _lower_quantile(values: Sequence[float], fraction: float = 0.05) -> float:
    if not values:
        raise ValueError("empty quantile")
    ordered = sorted(values)
    return ordered[int(math.floor(fraction * (len(ordered) - 1)))]


def monte_carlo(
    returns: Sequence[float], *, paths: int = 10_000, block_days: int = 21,
    seed: int = 20261102,
) -> MonteCarloResult:
    """Fixed-seed circular block bootstrap of complete daily net returns."""
    if not returns or paths <= 0 or block_days <= 0:
        raise ValueError("invalid Monte Carlo dimensions")
    values = tuple(float(ret) for ret in returns)
    n = len(values)
    rng = random.Random(seed)
    cagrs: list[float] = []
    sharpes: list[float] = []
    mdds: list[float] = []
    pfs: list[float] = []
    joint = 0
    for _ in range(paths):
        sample: list[float] = []
        while len(sample) < n:
            head = rng.randrange(n)
            count = min(block_days, n - len(sample))
            sample.extend(values[(head + offset) % n] for offset in range(count))
        stats = _returns_metrics(sample)
        cagrs.append(stats.cagr)
        sharpes.append(stats.sharpe if stats.sharpe is not None else -math.inf)
        mdds.append(stats.max_drawdown)
        pfs.append(stats.profit_factor)
        if stats.cagr >= 0.40 and stats.sharpe is not None and stats.sharpe > 1.5 and stats.max_drawdown >= -0.20 and stats.profit_factor > 1.3:
            joint += 1
    p05_cagr = _lower_quantile(cagrs)
    p05_sharpe = _lower_quantile(sharpes)
    return MonteCarloResult(paths, block_days, seed, p05_cagr,
                            p05_sharpe if math.isfinite(p05_sharpe) else None,
                            _lower_quantile(mdds), _lower_quantile(pfs),
                            joint / paths, joint / paths >= 0.05 and p05_cagr > 0)


def evaluate(
    result: SimulationResult, btc_daily_close: Mapping[date, float],
    *, mc_paths: int = 10_000,
) -> Evaluation:
    """Strict full Gate; vintage-less historical signals can never pass it."""
    metrics = summarize_returns(result)
    months = month_stability(result)
    folds = yearly_folds(result)
    regimes = regime_results(result, btc_daily_close)
    mc = monte_carlo([ret for _, ret in result.daily_returns], paths=mc_paths)
    expected = tuple(range(2021, 2026))
    full_calendar = result.start == datetime(2021, 1, 1, tzinfo=UTC) and result.end == datetime(2026, 1, 1, tzinfo=UTC) and metrics.observations == 1826
    numerical = (
        full_calendar and tuple(item.year for item in folds) == expected
        and metrics.cagr >= 0.40 and metrics.sharpe is not None and metrics.sharpe > 1.5
        and result.stress_drawdown >= -0.20 and metrics.profit_factor > 1.3
        and sum(item.passed for item in folds) >= 4 and months.passed and mc.passed
        and all(regime.days > 0 for regime in regimes)
        and not result.capacity_failed and not result.leverage_failed and not result.liquidated
    )
    return Evaluation(metrics, result.stress_drawdown, months, folds, regimes, mc,
                      numerical, numerical and result.independent_oos)


def run_registered_scenarios(
    adrs: Mapping[str, Mapping[date, tuple[float, datetime]]],
    trade4: Mapping[str, Mapping[datetime, object]],
    mark4: Mapping[str, Mapping[datetime, object]],
    mark1: Mapping[str, Mapping[datetime, object]],
    funding: Mapping[str, Mapping[datetime, float]],
    *, announcements: Mapping[str, datetime] | None = None,
    start: datetime = datetime(2021, 1, 1, tzinfo=UTC),
    end: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    symbols: Sequence[str] = SYMBOLS,
    strict_universe_bars: bool = True,
) -> dict[str, SimulationResult]:
    """Run 1x first; leverage only if its cost-after quality thresholds pass."""
    common = dict(announcements=announcements, start=start, end=end,
                  symbols=symbols, strict_universe_bars=strict_universe_bars)
    base = run_backtest(adrs, trade4, mark4, mark1, funding,
                        multiplier=1, slip_bps=15.0, **common)
    out = {"1x_15bp": base,
           "1x_25bp_stress": run_backtest(adrs, trade4, mark4, mark1, funding,
                                           multiplier=1, slip_bps=25.0, **common)}
    metrics = summarize_returns(base)
    if metrics.sharpe is not None and metrics.sharpe > 1.5 and metrics.profit_factor > 1.3:
        for mult in (3, 5):
            out[f"{mult}x_15bp"] = run_backtest(adrs, trade4, mark4, mark1, funding,
                                                multiplier=mult, slip_bps=15.0, **common)
            out[f"{mult}x_25bp_stress"] = run_backtest(adrs, trade4, mark4, mark1, funding,
                                                       multiplier=mult, slip_bps=25.0, **common)
    return out
