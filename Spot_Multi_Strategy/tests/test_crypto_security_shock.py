"""Synthetic timing and integrity checks; no formal history opened here."""

from datetime import date, datetime, timezone
import json

import pandas as pd

from research.crypto_security_shock_sim import simulate, weekly_target
from research import crypto_security_shock_source as source


def test_security_window_is_thirty_days_and_ends_eight_days_prior():
    monday = date(2021, 1, 4)
    assert weekly_target(monday, {monday.replace(day=3): 200_000_000}) == 1
    assert weekly_target(monday, {date(2020, 11, 27): 200_000_000}) == 1
    assert weekly_target(monday, {date(2020, 11, 28): 100_000_000}) == 0
    assert weekly_target(monday, {date(2020, 12, 27): 100_000_000}) == 0
    assert weekly_target(monday, {date(2020, 12, 28): 200_000_000}) == 1


def test_security_amounts_are_summed_and_equality_is_cash():
    monday = date(2021, 1, 4)
    assert weekly_target(monday, {date(2020, 12, 1): 40_000_000,
                                  date(2020, 12, 20): 60_000_000}) == 0
    assert weekly_target(monday, {date(2020, 12, 1): 40_000_000,
                                  date(2020, 12, 20): 59_999_999}) == 1
    try:
        weekly_target(date(2021, 1, 5), {})
    except ValueError:
        pass
    else:
        raise AssertionError("non-Monday decision accepted")


def test_source_rejects_unpinned_catalog_without_opening_price():
    body = json.dumps([{"date": int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()),
                        "name": "synthetic", "amount": 1}]).encode()
    try:
        source.parse_hacks(body)
    except source.SourceFailure as exc:
        assert "frozen hack catalog changed" in str(exc)
    else:
        raise AssertionError("unpinned catalog accepted")


def test_source_rejects_nonmidnight_or_negative_amount():
    stamp = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp())
    for row in ({"date": stamp + 1, "name": "synthetic", "amount": 1},
                {"date": stamp, "name": "synthetic", "amount": -1}):
        try:
            source.parse_hacks(json.dumps([row]).encode())
        except source.SourceFailure:
            pass
        else:
            raise AssertionError("invalid source row accepted")


def test_account_acts_first_monday_then_forces_terminal_sale():
    idx = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                           "close": 100.0, "quote_volume": 100_000_000.0}, index=idx)
    result = simulate({}, prices, 100.0)
    assert [x["day"] for x in result.fills] == ["2021-01-04", "2026-01-01"]
    assert result.terminal_cash < 10000.0
    delayed = simulate({}, prices, 100.0, latency_days=1)
    assert delayed.fills[0]["day"] == "2021-01-05"
