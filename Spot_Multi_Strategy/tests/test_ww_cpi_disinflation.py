"""Synthetic CPI timing/source/account checks; no formal CPI or BTC values opened."""

from datetime import date, datetime, timezone
from hashlib import sha256

import pandas as pd

from research import ww_cpi_disinflation_source as source
from research.ww_cpi_disinflation_sim import cpi_target, event_schedule, simulate


def _example_cpi() -> dict[date, float | None]:
    return {date(2019, 11, 1): 100.0, date(2019, 12, 1): 100.0,
            date(2020, 11, 1): 101.0, date(2020, 12, 1): 100.5}


def _event(month: date = date(2020, 12, 1)) -> dict:
    return {"released_at": datetime(2021, 1, 15, 13, 30, tzinfo=timezone.utc),
            "reference_month": month}


def test_yoy_deceleration_equality_and_missing_month():
    cpi = _example_cpi()
    assert cpi_target(date(2020, 12, 1), cpi) == 1
    cpi[date(2020, 12, 1)] = 101.0
    assert cpi_target(date(2020, 12, 1), cpi) == 0
    cpi[date(2020, 12, 1)] = 101.1
    assert cpi_target(date(2020, 12, 1), cpi) == 0
    cpi[date(2020, 12, 1)] = None
    assert cpi_target(date(2020, 12, 1), cpi) is None


def test_event_schedule_waits_for_release_and_carries_missing():
    event = _event()
    scheduled, unavailable = event_schedule((event,), _example_cpi(), 0)
    assert scheduled == {date(2021, 1, 16): 1} and unavailable == ()
    delayed, _ = event_schedule((event,), _example_cpi(), 1)
    assert delayed == {date(2021, 1, 17): 1}
    missing = _example_cpi(); missing[date(2020, 12, 1)] = None
    scheduled, unavailable = event_schedule((event,), missing, 0)
    assert scheduled == {} and unavailable == ("2021-01-15",)


def test_frozen_whalewatcher_release_manifest_integrity():
    events = source.release_manifest(source.EVENT_MANIFEST.read_bytes())
    assert len(events) == 71
    assert sum(2021 <= e["released_at"].year <= 2025 for e in events) == 59
    assert events[-1]["released_at"].date() == date(2025, 12, 18)


def _synthetic_fred(october_missing: bool = True) -> bytes:
    rows = ["observation_date,CPIAUCNS"]
    for period in pd.period_range("1913-01", "2026-09", freq="M"):
        day = period.to_timestamp().date()
        value = "." if day == date(2025, 10, 1) and october_missing else "100.0"
        rows.append(f"{day},{value}")
    return ("\n".join(rows) + "\n").encode()


def test_cpi_parser_detects_missing_month_mask_and_hash():
    body = _synthetic_fred()
    original = source.CPI_SHA
    try:
        source.CPI_SHA = sha256(body).hexdigest()
        values = source.parse_cpi(body)
        assert len(values) == 1365 and values[date(2025, 10, 1)] is None
        altered = _synthetic_fred(october_missing=False)
        source.CPI_SHA = sha256(altered).hexdigest()
        try:
            source.parse_cpi(altered)
        except source.SourceFailure as exc:
            assert "mask changed" in str(exc)
        else:
            raise AssertionError("unregistered CPI missing mask accepted")
    finally:
        source.CPI_SHA = original


def test_account_first_post_release_open_and_terminal_sale():
    index = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                           "close": 100.0, "quote_volume": 100_000_000.0}, index=index)
    primary = simulate((_event(),), _example_cpi(), prices, 100.0)
    assert [x["day"] for x in primary.fills] == ["2021-01-16", "2026-01-01"]
    assert primary.terminal_cash < 10000.0
    delayed = simulate((_event(),), _example_cpi(), prices, 100.0, latency_days=1)
    assert delayed.fills[0]["day"] == "2021-01-17"
