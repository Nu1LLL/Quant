"""Frozen spot execution ledger for the preregistered CME BTC COT study.

A return labelled D spans [D 00:00, D+1 00:00). The signal observed at
D 00:00 can only change the position at D+1 00:00. No market source is fetched
here; every bar and report comes from separately audited source loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
from statistics import median
from typing import Iterable

if __package__:
    from .cme_btc_cot import desired_exposure
else:
    from cme_btc_cot import desired_exposure


UTC = timezone.utc
START = datetime(2019, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, tzinfo=UTC)
WARMUP = datetime(2018, 12, 1, tzinfo=UTC)
INITIAL_EQUITY = 10_000.0
LOT_BTC = 0.00001
MIN_ORDER_USDT = 5.0
MAX_CAPACITY_RATIO = 0.01


class StudyDataError(ValueError):
    """A source or simulated execution is not valid for this frozen study."""


@dataclass(frozen=True)
class Trade:
    time: datetime
    side: str
    quantity_btc: float
    reference_price: float
    fill_price: float
    fee_usdt: float
    capacity_ratio: float


@dataclass(frozen=True)
class DailyRecord:
    day: date
    equity_open: float
    equity_next_open: float
    return_net: float
    exposed: bool
    conservative_high_equity: float
    conservative_low_equity: float


@dataclass(frozen=True)
class Ledger:
    records: tuple[DailyRecord, ...]
    trades: tuple[Trade, ...]
    final_equity: float
    conservative_max_drawdown: float
    capacity_pass: bool
    execution_pass: bool


def _spot_bars(bars: Iterable[object]) -> dict[datetime, object]:
    indexed: dict[datetime, object] = {}
    for bar in bars:
        stamp = getattr(bar, "open_time", None)
        if (not isinstance(stamp, datetime) or stamp.tzinfo is None or
                stamp.utcoffset() != timedelta(0) or stamp.time() != time.min):
            raise StudyDataError("spot bar must have a midnight UTC timestamp")
        if stamp in indexed:
            raise StudyDataError(f"duplicate spot bar: {stamp}")
        try:
            o, h, l, c = (float(getattr(bar, key)) for key in
                          ("open", "high", "low", "close"))
            qv = float(getattr(bar, "quote_volume"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise StudyDataError(f"non-numeric spot bar: {stamp}") from exc
        if (not all(math.isfinite(v) for v in (o, h, l, c, qv)) or
                min(o, h, l, c) <= 0 or qv < 0 or
                h < max(o, c, l) or l > min(o, c, h)):
            raise StudyDataError(f"invalid spot OHLC/volume: {stamp}")
        indexed[stamp] = bar
    cursor = WARMUP
    while cursor <= END:
        if cursor not in indexed:
            raise StudyDataError(f"missing daily spot bar: {cursor}")
        cursor += timedelta(days=1)
    return indexed


def _capacity_ratio(spot: dict[datetime, object], t: datetime,
                    notional: float) -> float:
    previous = [float(spot[t - timedelta(days=k)].quote_volume)
                for k in range(1, 31)]
    typical = median(previous)
    return notional / typical if typical > 0 else math.inf


def _records(spot: dict[datetime, object], equity: dict[datetime, float],
             exposure: dict[datetime, bool],
             extremes: dict[datetime, tuple[float, float]]) -> tuple[DailyRecord, ...]:
    records: list[DailyRecord] = []
    t = START
    while t < END:
        following = t + timedelta(days=1)
        opened, closed = equity[t], equity[following]
        if opened <= 0 or closed <= 0:
            raise StudyDataError("nonpositive account equity")
        high, low = extremes[t]
        records.append(DailyRecord(
            t.date(), opened, closed, closed / opened - 1.0,
            exposure[t], high, low,
        ))
        t = following
    return tuple(records)


def _drawdown(records: tuple[DailyRecord, ...]) -> float:
    peak = INITIAL_EQUITY
    worst = 0.0
    for row in records:
        peak = max(peak, row.equity_open, row.conservative_high_equity)
        worst = min(worst, row.conservative_low_equity / peak - 1.0,
                    row.equity_next_open / peak - 1.0)
        peak = max(peak, row.equity_next_open)
    return worst


def simulate(bars: Iterable[object], reports: Iterable[object], *,
             fee_bps: float = 10.0, slippage_bps: float = 10.0) -> Ledger:
    """Run the one fixed COT direction on verified daily BTC spot bars."""
    if fee_bps != 10.0 or slippage_bps not in (10.0, 25.0):
        raise ValueError("only preregistered main and 25bp stress costs allowed")
    spot = _spot_bars(bars)
    reports = tuple(reports)
    fee_rate = fee_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0
    cash = INITIAL_EQUITY
    quantity = 0.0
    pending: bool | None = None
    trades: list[Trade] = []
    equity: dict[datetime, float] = {}
    exposure: dict[datetime, bool] = {}
    extremes: dict[datetime, tuple[float, float]] = {}
    capacity_pass = execution_pass = True

    t = START
    while t <= END:
        opening = float(spot[t].open)
        target = False if t == END else pending
        if target is False and quantity > 0:
            fill = opening * (1.0 - slip_rate)
            notional = quantity * fill
            fee = notional * fee_rate
            if notional < MIN_ORDER_USDT:
                # Keep the full diagnostic ledger while failing the
                # executable-order Gate; do not silently drop liquidation.
                execution_pass = False
            ratio = _capacity_ratio(spot, t, notional)
            capacity_pass &= ratio <= MAX_CAPACITY_RATIO
            cash += notional - fee
            trades.append(Trade(t, "sell", quantity, opening, fill, fee, ratio))
            quantity = 0.0
        elif target is True and quantity == 0 and t < END:
            fill = opening * (1.0 + slip_rate)
            lots = math.floor(cash / (fill * (1.0 + fee_rate) * LOT_BTC) + 1e-12)
            bought = lots * LOT_BTC
            notional = bought * fill
            if notional < MIN_ORDER_USDT:
                execution_pass = False
            else:
                fee = notional * fee_rate
                ratio = _capacity_ratio(spot, t, notional)
                capacity_pass &= ratio <= MAX_CAPACITY_RATIO
                cash -= notional + fee
                if cash < -1e-7:
                    raise StudyDataError("BTC buy overspent cash")
                cash = max(0.0, cash)
                quantity = bought
                trades.append(Trade(t, "buy", bought, opening, fill, fee, ratio))

        marked = cash + quantity * opening
        if not math.isfinite(marked) or marked <= 0:
            raise StudyDataError("invalid marked account equity")
        equity[t] = marked
        exposure[t] = quantity > 0
        if t < END:
            bar = spot[t]
            extremes[t] = (cash + quantity * float(bar.high),
                           cash + quantity * float(bar.low))
            # The report observed now cannot affect the same opening price.
            desired = desired_exposure(reports, t)
            pending = desired if desired != (quantity > 0) else None
            if t + timedelta(days=1) == END:
                pending = None  # never buy solely for a terminal closeout
        t += timedelta(days=1)

    records = _records(spot, equity, exposure, extremes)
    return Ledger(records, tuple(trades), equity[END], _drawdown(records),
                  bool(capacity_pass), bool(execution_pass))


def buy_and_hold(bars: Iterable[object]) -> Ledger:
    """Passive BTC comparator with the same entry/exit cost and lot rule."""
    spot = _spot_bars(bars)
    first = float(spot[START].open)
    buy_fill = first * 1.001
    lots = math.floor(INITIAL_EQUITY / (buy_fill * 1.001 * LOT_BTC) + 1e-12)
    quantity = lots * LOT_BTC
    if quantity * buy_fill < MIN_ORDER_USDT:
        raise StudyDataError("passive benchmark cannot meet minimum order")
    buy_notional = quantity * buy_fill
    buy_fee = buy_notional * 0.001
    cash = INITIAL_EQUITY - buy_notional - buy_fee
    if cash < -1e-7:
        raise StudyDataError("passive benchmark overspent cash")
    cash = max(cash, 0.0)
    end_open = float(spot[END].open)
    sell_fill = end_open * 0.999
    sell_notional = quantity * sell_fill
    sell_fee = sell_notional * 0.001
    final = cash + sell_notional - sell_fee
    entry_ratio = _capacity_ratio(spot, START, buy_notional)
    exit_ratio = _capacity_ratio(spot, END, sell_notional)
    trades = (
        Trade(START, "buy", quantity, first, buy_fill, buy_fee, entry_ratio),
        Trade(END, "sell", quantity, end_open, sell_fill, sell_fee, exit_ratio),
    )
    equity: dict[datetime, float] = {}
    exposure: dict[datetime, bool] = {}
    extremes: dict[datetime, tuple[float, float]] = {}
    t = START
    while t < END:
        # The first pre-trade 10,000 USDT fixes the benchmark's entry-cost
        # accounting; all later opens represent the marked, invested account.
        equity[t] = (INITIAL_EQUITY if t == START else
                     cash + quantity * float(spot[t].open))
        exposure[t] = True
        extremes[t] = (cash + quantity * float(spot[t].high),
                       cash + quantity * float(spot[t].low))
        t += timedelta(days=1)
    equity[END] = final
    records = _records(spot, equity, exposure, extremes)
    return Ledger(records, trades, final, _drawdown(records),
                  entry_ratio <= MAX_CAPACITY_RATIO and
                  exit_ratio <= MAX_CAPACITY_RATIO,
                  sell_notional >= MIN_ORDER_USDT)
