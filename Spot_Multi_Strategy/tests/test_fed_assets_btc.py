"""Synthetic release-lag, source-integrity and account checks only."""

from datetime import date, timedelta
from hashlib import sha256

import pandas as pd

from research import fed_assets_btc_source as source
from research.fed_assets_btc_sim import simulate, weekly_target


def _synthetic_walcl() -> dict[date, float]:
    return {date(2020, 1, 1) + timedelta(days=7 * i): float(1_000_000 + i)
            for i in range(314)}


def test_monday_uses_twelve_day_lag_and_exact_four_week_change():
    walcl = _synthetic_walcl()
    monday = date(2021, 1, 4)
    assert weekly_target(monday, walcl) == 1
    # 2020-12-30 is five days old and must never affect the 2021-01-04 order.
    walcl[date(2020, 12, 30)] = 1.0
    assert weekly_target(monday, walcl) == 1
    walcl[date(2020, 12, 23)] = walcl[date(2020, 11, 25)]
    assert weekly_target(monday, walcl) == 0
    del walcl[date(2020, 12, 23)]
    try:
        weekly_target(monday, walcl)
    except source.SourceFailure:
        pass
    else:
        raise AssertionError("missing lagged Fed release accepted")


def test_unpinned_fred_csv_is_rejected_before_numeric_parse():
    try:
        source.parse_walcl(b"observation_date,WALCL\n2020-01-01,1\n")
    except source.SourceFailure as exc:
        assert "response changed" in str(exc)
    else:
        raise AssertionError("unpinned FRED response accepted")


def test_frozen_csv_schema_accepts_complete_synthetic_grid_only():
    body = "observation_date,WALCL\n" + "".join(
        f"{date(2020, 1, 1) + timedelta(days=7 * i)},{1_000_000+i}\n"
        for i in range(314))
    pin = source.WALCL_SHA
    source.WALCL_SHA = sha256(body.encode()).hexdigest()
    try:
        assert len(source.parse_walcl(body.encode())) == 314
    finally:
        source.WALCL_SHA = pin


def test_account_first_monday_and_terminal_liquidation():
    idx = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                           "close": 100.0, "quote_volume": 100_000_000.0}, index=idx)
    result = simulate(_synthetic_walcl(), prices, 100.0)
    assert [x["day"] for x in result.fills] == ["2021-01-04", "2026-01-01"]
    assert result.terminal_cash < 10_000
    delayed = simulate(_synthetic_walcl(), prices, 100.0, latency_days=1)
    assert delayed.fills[0]["day"] == "2021-01-05"
