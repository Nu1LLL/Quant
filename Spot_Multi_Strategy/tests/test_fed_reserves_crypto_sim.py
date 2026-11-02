"""Only synthetic fixtures; no Fed balance, crypto market, or PnL source is read."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research import fed_reserves_crypto_sim as sim
from research.fed_reserves_vintage_data import ReserveObservation


TEST_END = pd.Timestamp("2021-01-20 00:00:00", tz="UTC")
PRICES = {"BCHUSDT": 500.0, "BTCUSDT": 10_000.0,
          "DOGEUSDT": 0.1, "ETHUSDT": 1_000.0, "LTCUSDT": 100.0}


@pytest.fixture
def example(monkeypatch):
    monkeypatch.setattr(sim, "END", TEST_END)
    stamps = pd.date_range(sim.WARMUP, TEST_END, freq="4h", tz="UTC")
    settlements = pd.date_range(sim.WARMUP, TEST_END, freq="8h", tz="UTC")
    released = pd.to_datetime([
        "2020-11-19", "2020-11-26", "2020-12-03", "2020-12-10",
        "2020-12-17", "2020-12-24", "2020-12-31", "2021-01-07",
        "2021-01-14",
    ], utc=True)
    balances = [90.0, 90.0, 100.0, 100.0, 100.0, 110.0, 120.0, 80.0, 100.0]
    observations = [
        {"release_date": day, "observed_date": day - pd.Timedelta(days=1),
         "value": value}
        for day, value in zip(released, balances, strict=True)
    ]
    trade, mark, funding = {}, {}, {}
    days_since = np.asarray((stamps - sim.WARMUP) / pd.Timedelta(days=1), dtype=int)
    scale = np.where(stamps < sim.START, 1.0 + 0.01 * np.sin(days_since * 1.3), 1.0)
    for symbol in sim.SYMBOLS:
        price = PRICES[symbol] * scale
        trade[symbol] = pd.DataFrame({
            "open": price, "high": price, "low": price, "close": price,
            "quote_volume": np.full(len(stamps), 1_000_000.0),
        }, index=stamps)
        mark[symbol] = pd.DataFrame({
            "open": price, "high": price, "low": price, "close": price,
        }, index=stamps)
        for frame in (trade[symbol], mark[symbol]):
            frame.loc[TEST_END, ["high", "low", "close"]] = np.nan
        trade[symbol].loc[TEST_END, "quote_volume"] = np.nan
        funding[symbol] = pd.DataFrame({
            "funding_time": settlements,
            "raw_funding_time": settlements + pd.Timedelta(milliseconds=12),
            "rate": np.zeros(len(settlements)),
            "funding_interval_hours": np.full(len(settlements), 8.0),
        })
    return observations, trade, mark, funding


def _holdings_before(ledger, stamp):
    return {s: sum(t.quantity for t in ledger.trades
                   if t.symbol == s and pd.Timestamp(t.time) < stamp)
            for s in sim.SYMBOLS}


def test_four_releases_publication_lag_and_direction(example):
    observations, trade, mark, funding = example
    ledger = sim.run_backtest(observations, trade, mark, funding)
    first = pd.Timestamp("2021-01-04 00:00:00", tz="UTC")
    second = pd.Timestamp("2021-01-11 00:00:00", tz="UTC")
    third = pd.Timestamp("2021-01-18 00:00:00", tz="UTC")
    assert len(ledger.records) == 19
    assert ledger.records[0].day == date(2021, 1, 1)
    assert ledger.records[0].equity_open == 10_000.0
    assert not [t for t in ledger.trades if t.time < first]
    assert len([t for t in ledger.trades if t.time == first]) == 5
    assert all(t.side == "buy" for t in ledger.trades if t.time == first)
    assert all(_holdings_before(ledger, second)[s] > 0 for s in sim.SYMBOLS)
    assert all(_holdings_before(ledger, third)[s] < 0 for s in sim.SYMBOLS)
    assert all(_holdings_before(ledger, TEST_END)[s] == 0 for s in sim.SYMBOLS)
    assert ledger.terminal_flat
    assert ledger.fee_usdt > 0 and ledger.slippage_usdt > 0
    assert ledger.final_equity == pytest.approx(
        10_000 - ledger.fee_usdt - ledger.slippage_usdt)
    assert ledger.conservative_max_drawdown <= 0
    assert ledger.conservative_max_drawdown_by_year[2021] <= 0
    native = tuple(ReserveObservation(**row) for row in observations)
    assert sim.run_backtest(native, trade, mark, funding).final_equity == pytest.approx(
        ledger.final_equity)


def test_volatility_uses_only_prior_completed_days_and_caps_gross(example):
    observations, trade, mark, funding = example
    first = pd.Timestamp("2021-01-04 00:00:00", tz="UTC")
    index = int((first - sim.WARMUP) / pd.Timedelta(hours=4))
    checked = {s: {k: frame[k].to_numpy(dtype=float) for k in ("close",)}
               for s, frame in trade.items()}
    expected_vol = sim._realized_vol(checked, index // 6)
    assert expected_vol > 0
    ledger = sim.run_backtest(observations, trade, mark, funding)
    entries = [t for t in ledger.trades if t.time == first]
    expected_gross = min(3.0, 0.30 / expected_vol)
    observed_gross = sum(t.quantity * t.reference_price for t in entries) / 10_000
    assert 0 < observed_gross <= 3.0
    # Integer lots can only reduce, never increase, the target gross.
    assert observed_gross <= expected_gross + 1e-12
    assert expected_gross - observed_gross < 0.02
    # Even a dramatic change in the still-unfinished Monday bar is invisible
    # to the sizing calculation (although it would alter an actual trade fill).
    mutated = {s: {"close": data["close"].copy()} for s, data in checked.items()}
    for data in mutated.values():
        data["close"][index] = 1_000_000.0
    assert sim._realized_vol(mutated, index // 6) == pytest.approx(expected_vol)


def test_future_release_does_not_change_first_week_and_benchmark_is_same_risk(example):
    observations, trade, mark, funding = example
    first = pd.Timestamp("2021-01-04 00:00:00", tz="UTC")
    original = sim.run_backtest(observations, trade, mark, funding)
    revised = [dict(row) for row in observations]
    revised[7]["value"] = 1_000_000_000.0  # Jan 7 not known at Jan 4 decision.
    changed = sim.run_backtest(revised, trade, mark, funding)
    assert [(t.symbol, t.quantity) for t in original.trades if t.time == first] == [
        (t.symbol, t.quantity) for t in changed.trades if t.time == first]
    control = sim.run_backtest(observations, trade, mark, funding, benchmark=True)
    control_entries = [t for t in control.trades if t.time == first]
    assert len(control_entries) == 5
    assert all(t.side == "buy" for t in control_entries)
    assert control.terminal_flat
    assert control.final_equity == pytest.approx(
        10_000 - control.fee_usdt - control.slippage_usdt)
    for symbol in sim.SYMBOLS:
        rule_qty = next(t.quantity for t in original.trades
                        if t.time == first and t.symbol == symbol)
        control_qty = next(t.quantity for t in control_entries if t.symbol == symbol)
        assert rule_qty == control_qty


def test_friday_delayed_release_is_not_backdated_to_wednesday(example):
    observations, trade, mark, funding = example
    monday = pd.Timestamp("2021-01-04 00:00:00", tz="UTC")
    delayed = [dict(row) for row in observations]
    delayed[1]["value"] = 200.0
    delayed[6]["release_date"] = pd.Timestamp("2021-01-01", tz="UTC")
    # Although the observation is Wednesday Dec 30, its Friday Jan 1
    # publication fails the Monday-minus-four-day cutoff (Thu Dec 31).
    dates, values = sim._validate_observations(delayed)
    assert sim._direction(dates, values, monday) == -1
    ledger = sim.run_backtest(delayed, trade, mark, funding)
    assert all(t.side == "sell" for t in ledger.trades if t.time == monday)


def test_three_times_gross_cap_with_quiet_prior_returns(example):
    observations, trade, mark, funding = example
    for frames in (trade, mark):
        for symbol, frame in frames.items():
            early = frame.index < sim.START
            day_number = np.asarray((frame.index[early] - sim.WARMUP) /
                                    pd.Timedelta(days=1), dtype=int)
            mild = PRICES[symbol] * (1.0 + 0.0001 * np.sin(day_number * 1.3))
            for field in ("open", "high", "low", "close"):
                frame.loc[early, field] = mild
    first = pd.Timestamp("2021-01-04 00:00:00", tz="UTC")
    index = int((first - sim.WARMUP) / pd.Timedelta(hours=4))
    closes = {s: {"close": frame["close"].to_numpy(dtype=float)}
              for s, frame in trade.items()}
    assert 0.30 / sim._realized_vol(closes, index // 6) > 3.0
    ledger = sim.run_backtest(observations, trade, mark, funding)
    entries = [t for t in ledger.trades if t.time == first]
    gross = sum(t.quantity * t.reference_price for t in entries) / 10_000
    assert 2.98 <= gross <= 3.0
    assert ledger.terminal_flat


def test_old_holdings_pay_funding_before_rebalance(example):
    observations, trade, mark, funding = example
    second = pd.Timestamp("2021-01-11 00:00:00", tz="UTC")
    third = pd.Timestamp("2021-01-18 00:00:00", tz="UTC")
    for stamp, rate in ((second, 0.01), (third, 0.02)):
        for symbol in sim.SYMBOLS:
            funding[symbol].loc[funding[symbol].funding_time == stamp, "rate"] = rate
    ledger = sim.run_backtest(observations, trade, mark, funding)
    expected = sum(_holdings_before(ledger, second)[s] * PRICES[s] * 0.01 +
                   _holdings_before(ledger, third)[s] * PRICES[s] * 0.02
                   for s in sim.SYMBOLS)
    assert expected < 0  # The larger second settlement is received by old shorts.
    assert ledger.funding_usdt == pytest.approx(expected)
    assert ledger.final_equity == pytest.approx(
        10_000 - ledger.fee_usdt - ledger.slippage_usdt - expected)


def test_capacity_min_order_and_margin_are_fail_closed(example):
    observations, trade, mark, funding = example
    for frame in trade.values():
        frame.loc[frame.index < TEST_END, "quote_volume"] = 100.0
    ledger = sim.run_backtest(observations, trade, mark, funding)
    assert ledger.capacity_breaches >= 5
    for frame in (trade["DOGEUSDT"], mark["DOGEUSDT"]):
        for field in ("open", "high", "low", "close"):
            frame.loc[frame.index < TEST_END, field] = 10_000.0
        frame.loc[TEST_END, "open"] = 10_000.0
    control = sim.run_backtest(observations, trade, mark, funding, benchmark=True)
    assert control.execution_breaches >= 1
    assert not [t for t in control.trades if t.symbol == "DOGEUSDT"]
    # Restore a clean fixture and force an adverse all-leg event.
    for frame in (trade["DOGEUSDT"], mark["DOGEUSDT"]):
        for field in ("open", "high", "low", "close"):
            frame.loc[frame.index < TEST_END, field] = PRICES["DOGEUSDT"]
        frame.loc[TEST_END, "open"] = PRICES["DOGEUSDT"]
    for symbol in sim.SYMBOLS:
        stamp = pd.Timestamp("2021-01-05 04:00:00", tz="UTC")
        mark[symbol].loc[stamp, "low"] = PRICES[symbol] * 0.01
    with pytest.raises(sim.MarginRuinError, match="adverse 4h"):
        sim.run_backtest(observations, trade, mark, funding)


def test_malformed_vintages_gaps_and_terminal_future_bar_rejected(example):
    observations, trade, mark, funding = example
    bad = [dict(row) for row in observations]
    bad[6]["release_date"] = pd.Timestamp("2021-01-01", tz="UTC")
    bad[6]["observed_date"] = pd.Timestamp("2020-12-30", tz="UTC")
    assert sim._direction(*sim._validate_observations(bad),
                          pd.Timestamp("2021-01-04", tz="UTC")) == 1
    bad[6]["value"] = float("nan")
    with pytest.raises(sim.SimulationDataError, match="nonfinite"):
        sim.run_backtest(bad, trade, mark, funding)
    bad_mark = dict(mark)
    bad_mark["BTCUSDT"] = mark["BTCUSDT"].drop(pd.Timestamp("2021-01-08 04:00", tz="UTC"))
    with pytest.raises(sim.SimulationDataError, match="4h mark_bars"):
        sim.run_backtest(observations, trade, bad_mark, funding)
    bad_trade = dict(trade)
    bad_trade["LTCUSDT"] = trade["LTCUSDT"].copy()
    bad_trade["LTCUSDT"].loc[TEST_END, "close"] = 100.0
    with pytest.raises(sim.SimulationDataError, match="reveals future"):
        sim.run_backtest(observations, bad_trade, mark, funding)
    bad_funding = dict(funding)
    bad_funding["ETHUSDT"] = funding["ETHUSDT"].copy()
    bad_funding["ETHUSDT"].loc[0, "raw_funding_time"] += pd.Timedelta(milliseconds=101)
    with pytest.raises(sim.SimulationDataError, match="100ms"):
        sim.run_backtest(observations, trade, mark, bad_funding)


def test_zero_prior_basket_volatility_is_data_failure(example):
    observations, trade, mark, funding = example
    for frames in (trade, mark):
        for symbol, frame in frames.items():
            for field in ("open", "high", "low", "close"):
                frame.loc[frame.index < TEST_END, field] = PRICES[symbol]
    with pytest.raises(sim.SimulationDataError, match="zero prior-30-day"):
        sim.run_backtest(observations, trade, mark, funding)
