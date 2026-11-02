"""Synthetic-only tests for the frozen CME BTC spot execution ledger."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from Spot_Multi_Strategy.research.cme_btc_sim import (
    END, START, WARMUP, StudyDataError, buy_and_hold, simulate,
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


def synthetic_bars(*, quote_volume: float = 1_000_000_000.0) -> list[Bar]:
    out: list[Bar] = []
    day = WARMUP
    while day <= END:
        out.append(Bar(day, quote_volume=quote_volume))
        day += timedelta(days=1)
    return out


def test_no_report_signal_is_cash_and_has_complete_calendar():
    with patch("Spot_Multi_Strategy.research.cme_btc_sim.desired_exposure", return_value=False):
        ledger = simulate(synthetic_bars(), ())
    assert ledger.trades == ()
    assert ledger.final_equity == 10_000.0
    assert len(ledger.records) == (END - START).days == 2557
    assert all(not row.exposed and row.return_net == 0 for row in ledger.records)
    assert ledger.capacity_pass and ledger.execution_pass


def test_signal_enter_and_leave_only_at_next_open_with_costs():
    jan_10 = datetime(2019, 1, 10, tzinfo=UTC)
    with patch("Spot_Multi_Strategy.research.cme_btc_sim.desired_exposure",
               side_effect=lambda _reports, when: when < jan_10):
        ledger = simulate(synthetic_bars(), ())
    assert [(trade.time, trade.side) for trade in ledger.trades] == [
        (datetime(2019, 1, 2, tzinfo=UTC), "buy"),
        (datetime(2019, 1, 11, tzinfo=UTC), "sell"),
    ]
    assert sum(row.exposed for row in ledger.records) == 9
    assert ledger.final_equity < 10_000.0  # round trip costs on a flat market
    assert ledger.records[0].return_net < 0  # entry cost belongs to Jan 1 daily return
    assert ledger.records[9].return_net < 0  # exit cost belongs to Jan 10 daily return


def test_terminal_liquidation_and_passive_benchmark_count_both_fees():
    with patch("Spot_Multi_Strategy.research.cme_btc_sim.desired_exposure", return_value=True):
        ledger = simulate(synthetic_bars(), ())
    assert ledger.trades[0].time == datetime(2019, 1, 2, tzinfo=UTC)
    assert ledger.trades[-1].time == END
    assert ledger.trades[-1].side == "sell"
    assert not ledger.records[-1].return_net >= 0
    benchmark = buy_and_hold(synthetic_bars())
    assert benchmark.trades[0].time == START
    assert benchmark.trades[-1].time == END
    assert benchmark.final_equity < 10_000.0
    assert benchmark.records[0].equity_open == 10_000.0
    assert benchmark.records[-1].equity_next_open == benchmark.final_equity


def test_capacity_and_integrity_fail_closed_without_dropping_orders():
    with patch("Spot_Multi_Strategy.research.cme_btc_sim.desired_exposure", return_value=True):
        ledger = simulate(synthetic_bars(quote_volume=1_000.0), ())
    assert not ledger.capacity_pass
    assert len(ledger.trades) == 2
    bars = synthetic_bars()
    bars.pop(32)
    with pytest.raises(StudyDataError, match="missing daily spot bar"):
        simulate(bars, ())
    with pytest.raises(ValueError, match="only preregistered"):
        simulate(synthetic_bars(), (), slippage_bps=7.0)


def test_slippage_pressure_only_changes_frozen_execution_cost():
    with patch("Spot_Multi_Strategy.research.cme_btc_sim.desired_exposure", return_value=True):
        main = simulate(synthetic_bars(), ())
        stress = simulate(synthetic_bars(), (), slippage_bps=25.0)
    assert stress.final_equity < main.final_equity
    assert stress.conservative_max_drawdown <= main.conservative_max_drawdown
