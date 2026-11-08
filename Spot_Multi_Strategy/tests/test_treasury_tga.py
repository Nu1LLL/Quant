"""Synthetic TGA date-lag, phase and account checks; no formal balances opened."""

from datetime import date, timedelta

import pandas as pd

from research import treasury_tga_source as source
from research.treasury_tga_sim import decision_schedule, simulate, weekly_target


def test_tga_phase_fields_are_explicit():
    assert source._target(date(2021, 9, 30)) == ("Federal Reserve Account", "close_today_bal")
    assert source._target(date(2021, 10, 1)) == ("Treasury General Account (TGA)", "close_today_bal")
    assert source._target(date(2022, 4, 15)) == ("Treasury General Account (TGA)", "close_today_bal")
    assert source._target(date(2022, 4, 18)) == ("Treasury General Account (TGA) Closing Balance", "open_today_bal")


def test_tga_rule_uses_previous_monday_and_prior_four_weeks_only():
    monday = date(2021, 2, 8)
    values = {date(2021, 1, 4): 100.0, date(2021, 2, 1): 90.0}
    assert weekly_target(monday, values) == 1
    values[date(2021, 2, 2)] = 1e8
    assert weekly_target(monday, values) == 1
    values[date(2021, 2, 1)] = 100.0
    assert weekly_target(monday, values) == 0
    values[date(2021, 2, 1)] = 101.0
    assert weekly_target(monday, values) == 0


def test_tga_missing_and_five_day_fallback():
    monday = date(2021, 2, 8)
    assert weekly_target(monday, {}) is None
    assert weekly_target(monday, {date(2021, 1, 3): 100.0,
                                  date(2021, 1, 27): 90.0}) == 1
    assert weekly_target(monday, {date(2020, 12, 29): 100.0,
                                  date(2021, 2, 1): 90.0}) is None
    try:
        weekly_target(monday + timedelta(days=1), {})
    except ValueError:
        pass
    else:
        raise AssertionError("non-Monday decision accepted")


def test_source_rejects_unpinned_or_changed_treasury_payload():
    try:
        source.parse_tga(b"{}")
    except source.SourceFailure as exc:
        assert "changed" in str(exc)
    else:
        raise AssertionError("unregistered Treasury payload accepted")
    assert "record_date%3Agte%3A2021-01-01" in source.TGA_URL


def _synthetic_tga() -> dict[date, float]:
    days = pd.date_range("2021-01-04", "2025-12-31", freq="B")
    return {stamp.date(): 1_000_000.0 - i for i, stamp in enumerate(days)}


def test_account_starts_after_warmup_and_forces_terminal_sale():
    tga = _synthetic_tga()
    schedule, unavailable = decision_schedule(tga, 0)
    assert len(schedule) == 256 and len(unavailable) == 5
    assert schedule[date(2021, 2, 8)] == 1
    index = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                           "close": 100.0, "quote_volume": 100_000_000.0}, index=index)
    result = simulate(tga, prices, 100.0)
    assert [x["day"] for x in result.fills] == ["2021-02-08", "2026-01-01"]
    assert result.terminal_cash < 10000.0
    delayed = simulate(tga, prices, 100.0, latency_days=1)
    assert delayed.fills[0]["day"] == "2021-02-09"
