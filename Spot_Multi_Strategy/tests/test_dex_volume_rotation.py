"""Meaningful causality and adverse stop-fill checks with synthetic history."""

from datetime import date, timedelta
import math

import pandas as pd

from research.dex_volume_rotation_sim import (
    INITIAL, LOT, frozen_targets, simulate, weekly_target,
)


def _grid():
    return pd.date_range(date(2020, 12, 1), date(2025, 12, 31), tz="UTC", freq="D")


def _prices():
    frame = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                          "close": 100.0, "quote_volume": 1_000_000_000.0},
                         index=_grid())
    return {symbol: frame.copy() for symbol in ("BTCUSDT", *LOT)}


def test_sunday_volume_only_and_rank_tie_rule():
    volume = pd.DataFrame(100.0, index=_grid(), columns=sorted(LOT))
    thursday = date(2021, 1, 7)
    sunday = thursday - timedelta(days=4)
    days = pd.date_range(sunday - timedelta(days=13), sunday, tz="UTC", freq="D")
    volume.loc[days, "ETHUSDT"] = 200.0
    assert weekly_target(thursday, volume) == "ETHUSDT"
    volume.loc[pd.Timestamp(date(2021, 1, 5), tz="UTC"), "BNBUSDT"] = 1e12
    assert weekly_target(thursday, volume) == "ETHUSDT"


def test_stop_uses_prior_low_and_next_day_fill_then_reconciles():
    targets = {date(2021, 1, 7) + timedelta(days=7 * i): None for i in range(260)}
    targets[date(2021, 1, 7)] = "ETHUSDT"
    prices = _prices()
    prices["ETHUSDT"].loc[pd.Timestamp("2021-01-08", tz="UTC"), "low"] = 80.0
    prices["ETHUSDT"].loc[pd.Timestamp("2021-01-08", tz="UTC"), "close"] = 95.0
    prices["ETHUSDT"].loc[pd.Timestamp("2021-01-09", tz="UTC"), "open"] = 90.0
    prices["ETHUSDT"].loc[pd.Timestamp("2021-01-09", tz="UTC"), "low"] = 90.0
    terminal = {symbol: 100.0 for symbol in prices}
    result = simulate(targets, prices, terminal)
    buys = [x for x in result.fills if x["side"] == "buy"]
    stops = [x for x in result.fills if x["reason"] == "next_day_stop"]
    assert len(buys) == len(stops) == 1
    assert stops[0]["day"] == "2021-01-09"
    assert math.isclose(stops[0]["reference"], 80.0)
    assert math.isclose(stops[0]["execution"], 80.0 * .998)
    assert result.terminal_cash < INITIAL
    assert len(result.days) == 1826
    assert math.isclose(sum(day.end - day.start for day in result.days),
                        result.terminal_cash - INITIAL, abs_tol=1e-6)
