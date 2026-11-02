"""Synthetic-only tests; no market data or strategy outcomes are read here."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research import wiki_attention_sim as sim


TEST_END = pd.Timestamp("2021-01-15 00:00:00", tz="UTC")
PRICES = {"BCHUSDT": 500.0, "BTCUSDT": 10_000.0,
          "DOGEUSDT": 0.1, "ETHUSDT": 1_000.0, "LTCUSDT": 100.0}


@pytest.fixture
def example(monkeypatch):
    monkeypatch.setattr(sim, "END", TEST_END)
    days = pd.date_range(sim.WARMUP, TEST_END - pd.Timedelta(days=1),
                         freq="D", tz="UTC")
    bars = pd.date_range(sim.WARMUP, TEST_END, freq="4h", tz="UTC")
    settlements = pd.date_range(sim.WARMUP, TEST_END, freq="8h", tz="UTC")
    views = {}
    trade = {}
    mark = {}
    funding = {}
    for symbol in sim.SYMBOLS:
        page = pd.Series(100.0, index=days)
        if symbol == "BTCUSDT":
            page.loc["2020-12-28":"2021-01-03"] = 200.0
            page.loc["2021-01-04":"2021-01-10"] = 50.0
        if symbol == "BCHUSDT":
            page.loc["2020-12-28":"2021-01-03"] = 50.0
            page.loc["2021-01-04":"2021-01-10"] = 200.0
        views[symbol] = page
        price = PRICES[symbol]
        trade[symbol] = pd.DataFrame({
            "open": price, "high": price, "low": price, "close": price,
            "quote_volume": 1_000_000.0,
        }, index=bars)
        mark[symbol] = pd.DataFrame({
            "open": price, "high": price, "low": price, "close": price,
        }, index=bars)
        for frame in (trade[symbol], mark[symbol]):
            frame.loc[TEST_END, ["high", "low", "close"]] = np.nan
        trade[symbol].loc[TEST_END, "quote_volume"] = np.nan
        funding[symbol] = pd.DataFrame({
            "funding_time": settlements,
            "raw_funding_time": settlements + pd.Timedelta(milliseconds=10),
            "rate": np.zeros(len(settlements)),
            "funding_interval_hours": np.full(len(settlements), 8.0),
        })
    return views, trade, mark, funding


def _at(symbol, trades, timestamp):
    return [trade for trade in trades if trade.symbol == symbol and
            pd.Timestamp(trade.time) == pd.Timestamp(timestamp)]


def test_lag_direction_lots_terminal_and_cost_accounting(example):
    views, trade, mark, funding = example
    ledger = sim.run_backtest(views, trade, mark, funding)
    first = pd.Timestamp("2021-01-07 00:00:00", tz="UTC")
    second = pd.Timestamp("2021-01-14 00:00:00", tz="UTC")
    assert len(ledger.records) == 14
    assert ledger.records[0].day == date(2021, 1, 1)
    assert ledger.records[0].equity_open == 10_000.0
    assert not [t for t in ledger.trades if t.time < first]
    assert _at("BCHUSDT", ledger.trades, first)[0].side == "buy"
    assert _at("BTCUSDT", ledger.trades, first)[0].side == "sell"
    assert _at("BTCUSDT", ledger.trades, second)[-1].side == "buy"
    assert _at("BCHUSDT", ledger.trades, second)[-1].side == "sell"
    assert ledger.terminal_flat
    assert all(t.time <= TEST_END for t in ledger.trades)
    assert ledger.trades[-1].time == TEST_END
    assert ledger.fee_usdt > 0
    assert ledger.slippage_usdt > 0
    assert ledger.funding_usdt == 0.0
    assert ledger.final_equity == pytest.approx(
        10_000.0 - ledger.fee_usdt - ledger.slippage_usdt)
    assert ledger.conservative_max_drawdown <= 0
    assert ledger.conservative_max_drawdown_by_year[2021] <= 0


def test_no_lookahead_and_lexical_tie_break(example):
    views, trade, mark, funding = example
    original = sim.run_backtest(views, trade, mark, funding)
    modified = {symbol: series.copy() for symbol, series in views.items()}
    # The first decision is Jan 6 and may only see pageviews through Jan 3.
    modified["DOGEUSDT"].loc["2021-01-04":"2021-01-06"] = 1_000_000
    changed = sim.run_backtest(modified, trade, mark, funding)
    first = pd.Timestamp("2021-01-07 00:00:00", tz="UTC")
    original_first = [(t.symbol, t.side, t.quantity) for t in original.trades
                      if t.time == first]
    changed_first = [(t.symbol, t.side, t.quantity) for t in changed.trades
                     if t.time == first]
    assert changed_first == original_first
    all_flat = {symbol: pd.Series(100.0, index=series.index)
                for symbol, series in views.items()}
    # Equal negative scores choose the lexically first of BCH and BTC.
    all_flat["BCHUSDT"].loc["2020-12-28":"2021-01-03"] = 50
    all_flat["BTCUSDT"].loc["2020-12-28":"2021-01-03"] = 50
    target = sim._attention_target(all_flat, pd.Timestamp("2021-01-06", tz="UTC"))
    assert target["BCHUSDT"] == 0.5
    assert target["BTCUSDT"] == 0.0


def test_funding_at_rebalance_and_terminal_uses_old_signed_holdings(example):
    views, trade, mark, funding = example
    second = pd.Timestamp("2021-01-14 00:00:00", tz="UTC")
    terminal = TEST_END
    funding["BCHUSDT"].loc[
        funding["BCHUSDT"].funding_time == second, "rate"] = 0.01
    funding["BTCUSDT"].loc[
        funding["BTCUSDT"].funding_time == second, "rate"] = 0.02
    funding["BTCUSDT"].loc[
        funding["BTCUSDT"].funding_time == terminal, "rate"] = 0.01
    ledger = sim.run_backtest(views, trade, mark, funding)
    before_second = {symbol: sum(t.quantity for t in ledger.trades
                                 if t.symbol == symbol and t.time < second)
                     for symbol in sim.SYMBOLS}
    before_terminal = {symbol: sum(t.quantity for t in ledger.trades
                                   if t.symbol == symbol and t.time < terminal)
                       for symbol in sim.SYMBOLS}
    expected = (before_second["BCHUSDT"] * 500 * 0.01 +
                before_second["BTCUSDT"] * 10_000 * 0.02 +
                before_terminal["BTCUSDT"] * 10_000 * 0.01)
    assert ledger.funding_usdt == pytest.approx(expected)
    assert ledger.final_equity == pytest.approx(
        10_000 - ledger.fee_usdt - ledger.slippage_usdt - expected)


def test_benchmark_starts_jan_first_and_same_cost_model(example):
    views, trade, mark, funding = example
    ledger = sim.run_backtest(views, trade, mark, funding, benchmark=True)
    opening = [t for t in ledger.trades if t.time == sim.START]
    assert len(opening) == 5
    assert all(t.side == "buy" for t in opening)
    assert ledger.records[0].equity_open == 10_000.0
    assert ledger.records[0].equity_next_open < 10_000.0
    assert ledger.terminal_flat
    assert ledger.final_equity == pytest.approx(
        10_000.0 - ledger.fee_usdt - ledger.slippage_usdt)


def test_capacity_and_min_order_breaches_are_not_silently_ignored(example):
    views, trade, mark, funding = example
    for symbol in ("BCHUSDT", "BTCUSDT"):
        trade[symbol].loc[trade[symbol].index < TEST_END, "quote_volume"] = 100.0
    ledger = sim.run_backtest(views, trade, mark, funding)
    assert ledger.capacity_breaches >= 2
    # One DOGE lot at this synthetic price is too large for a 20% target.
    for frame in (trade["DOGEUSDT"], mark["DOGEUSDT"]):
        for column in ("open", "high", "low", "close"):
            frame.loc[frame.index < TEST_END, column] = 10_000.0
        frame.loc[TEST_END, "open"] = 10_000.0
    control = sim.run_backtest(views, trade, mark, funding, benchmark=True)
    assert control.execution_breaches >= 1
    assert not [t for t in control.trades if t.symbol == "DOGEUSDT"]


def test_ordered_four_hour_stress_and_margin_stop(example):
    views, trade, mark, funding = example
    stress_bar = pd.Timestamp("2021-01-08 04:00:00", tz="UTC")
    mark["BTCUSDT"].loc[stress_bar, "high"] = 14_000.0
    mark["BTCUSDT"].loc[stress_bar, "low"] = 9_000.0
    ledger = sim.run_backtest(views, trade, mark, funding)
    assert ledger.conservative_max_drawdown < -0.15
    assert ledger.conservative_max_drawdown_by_year[2021] == pytest.approx(
        ledger.conservative_max_drawdown)
    mark["BTCUSDT"].loc[stress_bar, "high"] = 40_000.0
    with pytest.raises(sim.MarginRuinError) as caught:
        sim.run_backtest(views, trade, mark, funding)
    assert caught.value.timestamp == stress_bar
    assert caught.value.stressed_equity <= 0
    assert "adverse 4h" in caught.value.reason


def test_stressed_maintenance_uses_adverse_notional_not_open_notional(example):
    views, trade, mark, funding = example
    stress_bar = pd.Timestamp("2021-01-08 04:00:00", tz="UTC")
    # A short BTC leg swells at its adverse high. The stressed account stays
    # above 5% of *opening* gross but falls below 5% of stressed gross.
    mark["BTCUSDT"].loc[stress_bar, "high"] = 28_300.0
    with pytest.raises(sim.MarginRuinError) as caught:
        sim.run_backtest(views, trade, mark, funding)
    assert caught.value.timestamp == stress_bar
    assert 500 < caught.value.stressed_equity < 1_000
    assert "maintenance proxy" in caught.value.reason


def test_missing_data_malformed_event_and_future_terminal_bar_fail_closed(example):
    views, trade, mark, funding = example
    bad_views = dict(views)
    bad_views["DOGEUSDT"] = views["DOGEUSDT"].copy()
    bad_views["DOGEUSDT"].loc["2021-01-03"] = 0
    with pytest.raises(sim.SimulationDataError, match="positive integers"):
        sim.run_backtest(bad_views, trade, mark, funding)
    bad_mark = dict(mark)
    bad_mark["BTCUSDT"] = mark["BTCUSDT"].drop(
        pd.Timestamp("2021-01-09 04:00:00", tz="UTC"))
    with pytest.raises(sim.SimulationDataError, match="4h mark_bars"):
        sim.run_backtest(views, trade, bad_mark, funding)
    bad_funding = dict(funding)
    bad_funding["ETHUSDT"] = funding["ETHUSDT"].copy()
    bad_funding["ETHUSDT"].loc[0, "raw_funding_time"] += pd.Timedelta(milliseconds=101)
    with pytest.raises(sim.SimulationDataError, match="100ms"):
        sim.run_backtest(views, trade, mark, bad_funding)
    bad_trade = dict(trade)
    bad_trade["LTCUSDT"] = trade["LTCUSDT"].copy()
    bad_trade["LTCUSDT"].loc[TEST_END, "high"] = 100.0
    with pytest.raises(sim.SimulationDataError, match="reveals future"):
        sim.run_backtest(views, bad_trade, mark, funding)
