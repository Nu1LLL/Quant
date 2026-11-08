"""Synthetic-only source, causal ranking, funding, and full replay checks."""

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
import zipfile

import pandas as pd

import research.defi_revenue_spread_data as archive_data
from research.defi_revenue_spread_data import FundingEvent, HourBar, SYMBOLS, TerminalOpen
from research.defi_revenue_spread_sim import (
    END, INPUT_START, START, _Account, evaluate, run_replay, weekly_rankings,
)
from research.defi_revenue_spread_source import parse_parent, RevenueSourceFailure


UTC = timezone.utc


def _revenue():
    dates = pd.date_range("2020-12-07", "2025-12-31", tz="UTC", freq="D")
    return pd.DataFrame({s: 100.0 for s in SYMBOLS}, index=dates)


def test_rank_uses_completed_sunday_and_fixed_tie_break():
    frame = _revenue()
    assert weekly_rankings(frame)[date(2021, 1, 7)] == (0, 2)
    sunday = pd.Timestamp("2021-01-03", tz="UTC")
    frame.loc[pd.date_range(pd.Timestamp("2020-12-28", tz="UTC"), sunday), "CRVUSDT"] = 1000.
    assert weekly_rankings(frame)[date(2021, 1, 7)] == (2, 1)
    frame.loc[pd.Timestamp("2021-01-04", tz="UTC"), "COMPUSDT"] = 1e12
    frame.loc[pd.Timestamp("2021-01-06", tz="UTC"), "COMPUSDT"] = 1e12
    assert weekly_rankings(frame)[date(2021, 1, 7)] == (2, 1)


def test_revenue_subset_hash_date_identity_and_nonnegative():
    rows = []
    for day in pd.date_range("2020-12-07", "2025-12-31", tz="UTC", freq="D"):
        rows.append([int(day.timestamp()), 100])
    digest = sha256(json.dumps(rows, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    parent = ("AAVEUSDT", "aave", "parent#aave", "AAVE", digest)
    body = json.dumps({"defillamaId": "parent#aave", "symbol": "AAVE",
                       "totalDataChart": rows}).encode()
    values, audit = parse_parent(body, parent)
    assert len(values) == 1851 and values[-1] == 100
    assert audit["subset_sha256"] == digest
    altered = rows.copy()
    altered[0] = [altered[0][0], -1]
    corrupt = json.dumps({"defillamaId": "parent#aave", "symbol": "AAVE",
                          "totalDataChart": altered}).encode()
    try:
        parse_parent(corrupt, parent)
    except RevenueSourceFailure:
        pass
    else:
        raise AssertionError("changed revenue history accepted")


def test_pair_costs_lot_and_funding_sign():
    account = _Account("synthetic", .001, .0005, 0)
    account.begin_day(date(2021, 1, 7))
    prices = (100., 50., 1.)
    account.enter(date(2021, 1, 7), prices, 0, 2, (1e9, 1e9, 1e9))
    assert account.position is not None
    assert math.isclose(account.position.long_qty / .1,
                        round(account.position.long_qty / .1), abs_tol=1e-8)
    before = account.cash
    account.fund(FundingEvent(datetime(2021, 1, 7, 8, tzinfo=UTC), "AAVEUSDT", .0001),
                 prices)
    assert account.cash < before
    before = account.cash
    account.fund(FundingEvent(datetime(2021, 1, 7, 8, tzinfo=UTC), "CRVUSDT", .0001),
                 prices)
    assert account.cash > before
    account.close(date(2021, 1, 14), prices)
    assert account.cash < 10000
    assert len(account.trades) == 1
    assert math.isclose(account.trades[0]["funding_pnl"], account.total_funding)


def test_terminal_reads_only_first_january_hour_and_funding():
    stamp = int(END.timestamp() * 1000)
    price = BytesIO()
    with zipfile.ZipFile(price, "w") as archive:
        archive.writestr("AAVEUSDT-1h-2026-01.csv",
                         f"{stamp},100,101,99,100,1,{stamp+3600000-1},100,1,1,1,0\n"
                         "future row intentionally malformed")
    funding = BytesIO()
    with zipfile.ZipFile(funding, "w") as archive:
        archive.writestr("AAVEUSDT-fundingRate-2026-01.csv",
                         "calc_time,funding_interval_hours,last_funding_rate\n"
                         f"{stamp},8,0.0001\nfuture row intentionally malformed")
    real = archive_data._verified_zip
    try:
        archive_data._verified_zip = lambda _symbol, _month, kind, _pin: zipfile.ZipFile(
            BytesIO(price.getvalue() if kind == "1h" else funding.getvalue()))
        assert archive_data._terminal_month("AAVEUSDT", "2026-01", "ignored") == 100.
        events = archive_data._funding_month("AAVEUSDT", "2026-01", "ignored")
        assert len(events) == 1 and events[0].rate == .0001
    finally:
        archive_data._verified_zip = real


def _synthetic_bars():
    prices = (100., 50., 1.)
    for stamp in pd.date_range(INPUT_START, END - timedelta(hours=1),
                               tz="UTC", freq="h"):
        yield HourBar(stamp.to_pydatetime(), prices, prices, prices, prices,
                      (1e6, 1e6, 1e6))
    yield TerminalOpen(END, prices)


def _synthetic_funding():
    time = START
    while time <= END:
        for symbol in SYMBOLS:
            yield FundingEvent(time, symbol, 0.0)
        time += timedelta(hours=8)


def test_five_year_hourly_account_and_gate_serialize():
    accounts, states, audit = run_replay(_revenue(), _synthetic_bars(),
                                         _synthetic_funding())
    assert audit["weekly_decisions"] == 260
    assert audit["funding_events"] == 16437
    assert len(accounts["net"].days) == 1826
    assert len(accounts["net"].trades) == 1
    assert accounts["net"].trades[0]["entry_day"] == "2021-01-07"
    assert accounts["net"].trades[0]["exit_day"] == "2026-01-01"
    assert accounts["gross_no_cost"].cash == 10000
    assert accounts["net"].cash < 10000
    result = evaluate(accounts, states, audit)
    assert result["monthly"]["months"] == 60
    assert result["monte_carlo"]["paths"] == 5000
    assert result["checks"]["independent_executable_oos"] is False
    assert result["all_gate_passed"] is False
    json.dumps(result, allow_nan=False)
