"""Synthetic-only source identity, causal signal and executable account checks."""

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math

import pandas as pd

from research.pow_issuance_data import FundingEvent, HourBar, SYMBOLS, TerminalOpen
from research.pow_issuance_sim import (
    END, INPUT_START, START, _Account, evaluate, run_replay, weekly_rankings,
)
from research.pow_issuance_source import IssuanceSourceFailure, parse_asset


UTC = timezone.utc


def _native_frame():
    days = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
    columns = {}
    for symbol, minted in zip(SYMBOLS, (10., 20., 30.)):
        columns[(symbol, "IssTotNtv")] = [minted] * len(days)
        columns[(symbol, "SplyCur")] = [1000.] * len(days)
    return pd.DataFrame(columns, index=days)


def test_sunday_issuance_is_causal_and_low_dilution_is_long():
    frame = _native_frame()
    assert weekly_rankings(frame)[date(2021, 1, 7)] == (0, 2)
    sunday = pd.Timestamp("2021-01-03", tz="UTC")
    frame.loc[pd.date_range(pd.Timestamp("2020-12-07", tz="UTC"), sunday),
              ("BCHUSDT", "IssTotNtv")] = 100.
    assert weekly_rankings(frame)[date(2021, 1, 7)] == (1, 0)
    frame.loc[pd.date_range("2021-01-04", "2021-01-06", tz="UTC"),
              ("ETCUSDT", "IssTotNtv")] = 1000.
    assert weekly_rankings(frame)[date(2021, 1, 7)] == (1, 0)


def test_source_hash_and_exact_midnight_grid():
    rows = []
    for day in pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D"):
        rows.append({"asset": "bch", "time": day.strftime("%Y-%m-%dT00:00:00.000000000Z"),
                     "IssTotNtv": "10", "SplyCur": "1000"})
    digest = sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode()).hexdigest()
    fixture = ("BCHUSDT", "bch", digest)
    values, audit = parse_asset(json.dumps({"data": rows}).encode(), fixture)
    assert len(values[("BCHUSDT", "IssTotNtv")]) == 1857
    assert audit["canonical_data_sha256"] == digest
    rows[0]["time"] = "2020-12-02T00:00:00.000000000Z"
    try:
        parse_asset(json.dumps({"data": rows}).encode(), fixture)
    except IssuanceSourceFailure:
        pass
    else:
        raise AssertionError("changed source history accepted")


def test_pair_fee_lot_and_funding_sign():
    account = _Account("synthetic", .001, .0005, 0)
    account.begin_day(date(2021, 1, 7))
    prices = (100., 1., 50.)
    account.enter(date(2021, 1, 7), prices, 0, 2, (1e9, 1e9, 1e9))
    assert account.position is not None
    assert math.isclose(account.position.long_qty / .1,
                        round(account.position.long_qty / .1), abs_tol=1e-8)
    before = account.cash
    account.fund(FundingEvent(datetime(2021, 1, 7, 8, tzinfo=UTC), "BCHUSDT", .0001), prices)
    assert account.cash < before
    before = account.cash
    account.fund(FundingEvent(datetime(2021, 1, 7, 8, tzinfo=UTC), "ETCUSDT", .0001), prices)
    assert account.cash > before
    account.close(date(2021, 1, 14), prices)
    assert account.cash < 10000
    assert math.isclose(account.trades[0]["funding_pnl"], account.total_funding)


def _flat_bars():
    prices = (100., 1., 50.)
    for stamp in pd.date_range(INPUT_START, END - timedelta(hours=1), tz="UTC", freq="h"):
        yield HourBar(stamp.to_pydatetime(), prices, prices, prices, prices,
                      (1e6, 1e6, 1e6))
    yield TerminalOpen(END, prices)


def _zero_funding():
    time = START
    while time <= END:
        for symbol in SYMBOLS:
            yield FundingEvent(time, symbol, 0.0)
        time += timedelta(hours=8)


def test_five_year_account_and_full_gate_on_flat_synthetic_market():
    accounts, states, audit = run_replay(_native_frame(), _flat_bars(), _zero_funding())
    assert audit["weekly_decisions"] == 260
    assert audit["funding_events"] == 16437
    assert len(accounts["net"].days) == 1826
    assert len(accounts["net"].trades) == 1
    assert accounts["gross_no_cost"].cash == 10000
    assert accounts["net"].cash < 10000
    result = evaluate(accounts, states, audit)
    assert result["monthly"]["months"] == 60
    assert result["monte_carlo"]["paths"] == 5000
    assert result["checks"]["independent_executable_oos"] is False
    assert result["all_gate_passed"] is False
    json.dumps(result, allow_nan=False)
