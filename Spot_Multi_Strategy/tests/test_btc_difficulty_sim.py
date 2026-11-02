from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from Spot_Multi_Strategy.research.btc_difficulty_sim import (
    END, START, StudyDataError, _midnight_after, buy_and_hold, simulate,
)


UTC = timezone.utc


@dataclass(frozen=True)
class Bar:
    open_time: datetime
    open: float = 100.0
    high: float = 102.0
    low: float = 98.0
    close: float = 100.0
    quote_volume: float = 1_000_000_000.0


@dataclass(frozen=True)
class Event:
    height: int
    timestamp: datetime
    knowledge_time: datetime
    recovery: bool


def bars(*, quote_volume=1_000_000_000.0):
    day = START - timedelta(days=31)
    out = []
    while day <= END:
        out.append(Bar(day, quote_volume=quote_volume))
        day += timedelta(days=1)
    return out


def test_knowledge_is_strictly_later_midnight():
    assert _midnight_after(datetime(2018, 1, 3, 0, tzinfo=UTC)) == datetime(
        2018, 1, 4, tzinfo=UTC)
    assert _midnight_after(datetime(2018, 1, 3, 23, tzinfo=UTC)) == datetime(
        2018, 1, 4, tzinfo=UTC)


def test_recovery_enters_one_day_after_decision_then_exits_after_28_days():
    event = Event(2016 * 250, datetime(2018, 1, 1, tzinfo=UTC),
                  datetime(2018, 1, 3, 12, tzinfo=UTC), True)
    ledger = simulate(bars(), [event])
    assert len(ledger.trades) == 2
    assert ledger.trades[0].time == datetime(2018, 1, 5, tzinfo=UTC)
    assert ledger.trades[1].time == datetime(2018, 2, 2, tzinfo=UTC)
    assert ledger.trades[0].side == "buy"
    assert ledger.trades[1].side == "sell"
    assert ledger.final_equity < 10_000
    assert ledger.capacity_pass
    assert len(ledger.records) == (END - START).days
    assert sum(rec.exposed for rec in ledger.records) == 28


def test_events_while_held_do_not_extend_or_open_new_position():
    first = Event(2016 * 250, datetime(2018, 1, 1, tzinfo=UTC),
                  datetime(2018, 1, 3, 12, tzinfo=UTC), True)
    second = Event(2016 * 251, datetime(2018, 1, 8, tzinfo=UTC),
                   datetime(2018, 1, 10, tzinfo=UTC), True)
    ledger = simulate(bars(), [first, second])
    assert len(ledger.trades) == 2
    assert ledger.ignored_events_while_held == 1


def test_warmup_event_is_not_traded_and_no_signal_is_all_cash():
    warmup = Event(2016 * 250, datetime(2017, 12, 31, tzinfo=UTC),
                   datetime(2018, 1, 2, tzinfo=UTC), True)
    ledger = simulate(bars(), [warmup])
    assert ledger.trades == ()
    assert ledger.final_equity == 10_000
    assert ledger.conservative_max_drawdown == 0
    assert all(record.return_net == 0 for record in ledger.records)


def test_capacity_failure_preserves_diagnostic_orders():
    event = Event(2016 * 250, datetime(2018, 1, 1, tzinfo=UTC),
                  datetime(2018, 1, 3, tzinfo=UTC), True)
    ledger = simulate(bars(quote_volume=1_000), [event])
    assert not ledger.capacity_pass
    assert len(ledger.trades) == 2
    assert ledger.trades[0].capacity_ratio > 0.01


def test_gapped_spot_day_and_unregistered_cost_fail_closed():
    source = bars()
    source.pop(40)
    with pytest.raises(StudyDataError, match="missing spot daily bar"):
        simulate(source, [])
    with pytest.raises(ValueError, match="only the preregistered"):
        simulate(bars(), [], slippage_bps=7.0)


def test_passive_benchmark_includes_entry_and_exit_cost():
    baseline = buy_and_hold(bars())
    assert baseline.final_equity < 10_000
    assert baseline.records[0].equity_open == 10_000
    assert baseline.records[0].return_net < 0
    assert len(baseline.trades) == 2
