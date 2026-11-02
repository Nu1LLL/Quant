"""Frozen, long-only ledger for the preregistered BTC difficulty-recovery study.

No source is fetched here.  A daily return labelled D spans [D 00:00, D+1
00:00); orders due at D+1 are included in that return.  All BTC data are
provided by the separately audited spot archive loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
from statistics import median
from typing import Iterable


UTC = timezone.utc
START = datetime(2018, 1, 1, tzinfo=UTC)
END = datetime(2026, 10, 1, tzinfo=UTC)
INITIAL_EQUITY = 10_000.0
LOT_BTC = 0.00001
MIN_ORDER_USDT = 5.0
CAPACITY_FRACTION = 0.01


class StudyDataError(ValueError):
    """An input is insufficient to calculate the sealed strategy fairly."""


@dataclass(frozen=True)
class Trade:
    time: datetime
    side: str
    quantity_btc: float
    reference_price: float
    fill_price: float
    fee_usdt: float
    capacity_ratio: float
    source_height: int | None


@dataclass(frozen=True)
class DailyRecord:
    day: date
    equity_open: float
    equity_next_open: float
    return_net: float
    exposed: bool
    conservative_drawdown: float
    conservative_high_equity: float
    conservative_low_equity: float
    regime_reference_close: float | None


@dataclass(frozen=True)
class Ledger:
    records: tuple[DailyRecord, ...]
    trades: tuple[Trade, ...]
    final_equity: float
    conservative_max_drawdown: float
    capacity_pass: bool
    ignored_events_while_held: int
    eligible_recovery_events: int


def _midnight_after(instant: datetime) -> datetime:
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise StudyDataError("knowledge time must be timezone-aware UTC")
    return datetime.combine(instant.date() + timedelta(days=1), time(0), UTC)


def _bars_by_time(bars: Iterable[object]) -> dict[datetime, object]:
    out: dict[datetime, object] = {}
    for bar in bars:
        stamp = bar.open_time
        if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise StudyDataError("non-UTC spot daily bar")
        if stamp.time() != time(0):
            raise StudyDataError(f"off-grid spot daily bar: {stamp}")
        if stamp in out:
            raise StudyDataError(f"duplicate spot daily bar: {stamp}")
        o, h, l, c = (float(getattr(bar, key)) for key in ("open", "high", "low", "close"))
        qv = float(bar.quote_volume)
        if (not all(math.isfinite(x) for x in (o, h, l, c, qv)) or
                min(o, h, l, c) <= 0 or qv < 0 or h < max(o, c, l) or l > min(o, c, h)):
            raise StudyDataError(f"invalid spot OHLC/quote volume: {stamp}")
        out[stamp] = bar
    required = START - timedelta(days=31)
    while required <= END:
        if required not in out:
            raise StudyDataError(f"missing spot daily bar: {required}")
        required += timedelta(days=1)
    return out


def _decisions(events: Iterable[object]) -> dict[datetime, tuple[object, ...]]:
    by_time: dict[datetime, list[object]] = {}
    heights: set[int] = set()
    previous_height: int | None = None
    for event in events:
        height = int(event.height)
        if height % 2016 or height in heights or (previous_height is not None and height <= previous_height):
            raise StudyDataError("difficulty events must be unique, ordered retarget heights")
        heights.add(height)
        previous_height = height
        if not bool(event.recovery):
            continue
        if event.timestamp.tzinfo is None or event.timestamp.utcoffset() != timedelta(0):
            raise StudyDataError("non-UTC retarget block timestamp")
        # Blocks before the frozen test window only warm up prior adjustments.
        if event.timestamp < START:
            continue
        decision = _midnight_after(event.knowledge_time)
        if START <= decision < END:
            by_time.setdefault(decision, []).append(event)
    return {key: tuple(value) for key, value in by_time.items()}


def simulate(bars: Iterable[object], events: Iterable[object], *,
             fee_bps: float = 10.0, slippage_bps: float = 10.0) -> Ledger:
    """Run exactly one fixed 2018-01-01..2026-10-01 spot-only scenario.

    The alternative 25bp slippage run changes only the frozen cost stress;
    it is not an adaptive parameter search.  An order over the capacity limit
    remains in this diagnostic ledger and marks capacity_pass=False.
    """
    if fee_bps != 10.0 or slippage_bps not in (10.0, 25.0):
        raise ValueError("only the preregistered main/stress costs are allowed")
    spot = _bars_by_time(bars)
    decisions = _decisions(events)
    cash = INITIAL_EQUITY
    quantity = 0.0
    exit_time: datetime | None = None
    planned_entry: tuple[datetime, int] | None = None
    trades: list[Trade] = []
    equity_at: dict[datetime, float] = {}
    exposure_at: dict[datetime, bool] = {}
    pressure_at: dict[datetime, float] = {}
    pressure_extremes: dict[datetime, tuple[float, float]] = {}
    high_water = INITIAL_EQUITY
    worst_dd = 0.0
    capacity_pass = True
    ignored = eligible = 0
    fee_rate = fee_bps / 10_000
    slip_rate = slippage_bps / 10_000

    def capacity(t: datetime, notional: float) -> float:
        previous = [float(spot[t - timedelta(days=k)].quote_volume)
                    for k in range(1, 31)]
        traded = median(previous)
        if traded <= 0:
            return math.inf
        return notional / traded

    t = START
    while t <= END:
        bar = spot[t]
        opening = float(bar.open)
        # Determine event eligibility before executing any order at this same
        # midnight: an event during a position closing now remains ignored.
        decisions_now = decisions.get(t, ())
        for event in decisions_now:
            eligible += 1
            if quantity > 0 or planned_entry is not None:
                ignored += 1
                continue
            candidate = t + timedelta(days=1)
            if candidate < END:
                planned_entry = (candidate, int(event.height))

        if quantity > 0 and (exit_time == t or t == END):
            fill = opening * (1 - slip_rate)
            notional = quantity * fill
            fee = notional * fee_rate
            ratio = capacity(t, notional)
            capacity_pass &= ratio <= CAPACITY_FRACTION
            cash += notional - fee
            trades.append(Trade(t, "sell", quantity, opening, fill, fee, ratio, None))
            quantity = 0.0
            exit_time = None

        if planned_entry is not None and planned_entry[0] == t and t < END:
            if quantity != 0:
                raise StudyDataError("overlapping recovery positions")
            fill = opening * (1 + slip_rate)
            size = math.floor((cash / (fill * (1 + fee_rate))) / LOT_BTC + 1e-12) * LOT_BTC
            notional = size * fill
            if notional < MIN_ORDER_USDT:
                capacity_pass = False
                size = 0.0
            if size > 0:
                fee = notional * fee_rate
                ratio = capacity(t, notional)
                capacity_pass &= ratio <= CAPACITY_FRACTION
                cash -= notional + fee
                if cash < -1e-7:
                    raise StudyDataError("entry overspent USDT cash")
                cash = max(cash, 0.0)
                quantity = size
                exit_time = min(t + timedelta(days=28), END)
                trades.append(Trade(t, "buy", size, opening, fill, fee, ratio,
                                    planned_entry[1]))
            planned_entry = None

        equity = cash + quantity * opening
        if equity < 0 or not math.isfinite(equity):
            raise StudyDataError("invalid account equity")
        equity_at[t] = equity
        exposure_at[t] = quantity > 0
        high_water = max(high_water, equity)
        worst_dd = min(worst_dd, equity / high_water - 1)
        if t < END:
            peak_today = cash + quantity * float(bar.high)
            trough_today = cash + quantity * float(bar.low)
            high_water = max(high_water, peak_today)
            worst_dd = min(worst_dd, trough_today / high_water - 1)
            pressure_at[t] = worst_dd
            pressure_extremes[t] = (peak_today, trough_today)
        t += timedelta(days=1)

    records: list[DailyRecord] = []
    t = START
    while t < END:
        next_t = t + timedelta(days=1)
        prior = equity_at[t]
        current = equity_at[next_t]
        if prior <= 0:
            raise StudyDataError("nonpositive equity before terminal")
        records.append(DailyRecord(
            day=t.date(), equity_open=prior, equity_next_open=current,
            return_net=current / prior - 1, exposed=exposure_at[t],
            conservative_drawdown=pressure_at[t],
            conservative_high_equity=pressure_extremes[t][0],
            conservative_low_equity=pressure_extremes[t][1],
            regime_reference_close=(float(spot[t - timedelta(days=1)].close)
                                    if t - timedelta(days=31) in spot else None),
        ))
        t = next_t
    return Ledger(tuple(records), tuple(trades), equity_at[END], worst_dd,
                  capacity_pass, ignored, eligible)


def buy_and_hold(bars: Iterable[object]) -> Ledger:
    """Same-period BTC spot benchmark, same fee/slippage and lot rule.

    This is not a second Alpha hypothesis: it is a fixed passive comparator.
    """
    spot = _bars_by_time(bars)
    initial = INITIAL_EQUITY
    first = float(spot[START].open)
    fill_in = first * 1.001
    quantity = math.floor((initial / (fill_in * 1.001)) / LOT_BTC + 1e-12) * LOT_BTC
    cash = initial - quantity * fill_in * 1.001
    last = float(spot[END].open)
    fill_out = last * 0.999
    final = cash + quantity * fill_out * 0.999
    records: list[DailyRecord] = []
    t = START
    high_water = initial
    worst_dd = 0.0
    while t < END:
        current = cash + quantity * float(spot[t].open)
        if t == START:
            # Count the benchmark's initial fill and fee in the first day's
            # return; otherwise passive BTC gets a free entry.
            current = INITIAL_EQUITY
        next_t = t + timedelta(days=1)
        upcoming = final if next_t == END else cash + quantity * float(spot[next_t].open)
        high_water = max(high_water, cash + quantity * float(spot[t].high))
        worst_dd = min(worst_dd,
                       (cash + quantity * float(spot[t].low)) / high_water - 1,
                       upcoming / high_water - 1)
        records.append(DailyRecord(t.date(), current, upcoming,
                                   upcoming / current - 1, True, worst_dd,
                                   cash + quantity * float(spot[t].high),
                                   cash + quantity * float(spot[t].low),
                                   float(spot[t - timedelta(days=1)].close)))
        t = next_t
    entry_notional = quantity * fill_in
    exit_notional = quantity * fill_out
    return Ledger(tuple(records), (
        Trade(START, "buy", quantity, first, fill_in, entry_notional * 0.001,
              math.nan, None),
        Trade(END, "sell", quantity, last, fill_out, exit_notional * 0.001,
              math.nan, None),
    ), final, worst_dd, True, 0, 0)
