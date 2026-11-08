"""Synthetic ratio, archive and account checks; no historical values opened."""

from datetime import date, datetime, timedelta, timezone
from io import BytesIO
import zipfile

import pandas as pd

from research.btc_perp_spot_turnover_sim import simulate, weekly_target
from research.btc_perp_spot_turnover_source import SourceFailure, _parse_futures_zip, HEADER


def _window(monday: date) -> tuple[dict[date, float], dict[date, float]]:
    days = [monday - timedelta(days=k) for k in range(56, 0, -1)]
    return ({x: 100.0 for x in days},
            {x: 200.0 if i < 28 else 100.0 for i, x in enumerate(days)})


def test_ratio_uses_two_nonoverlapping_completed_windows():
    monday = date(2021, 1, 4)
    spot, futures = _window(monday)
    assert weekly_target(monday, spot, futures) == 1
    futures[monday - timedelta(days=1)] = 2900.0
    assert weekly_target(monday, spot, futures) == 0
    futures[monday - timedelta(days=1)] = 100.0
    futures[monday] = 1e12
    assert weekly_target(monday, spot, futures) == 1
    futures = {x: 100.0 for x in spot}
    assert weekly_target(monday, spot, futures) == 0
    assert weekly_target(monday, spot, {x: 200.0 for x in spot}) == 0


def test_ratio_rejects_missing_invalid_and_non_monday():
    monday = date(2021, 1, 4)
    spot, futures = _window(monday)
    spot.pop(monday - timedelta(days=56))
    for s, f in ((spot, futures), (*_window(monday),)):
        if len(s) == 56:
            f[monday - timedelta(days=29)] = 0.0
        try:
            weekly_target(monday, s, f)
        except SourceFailure:
            pass
        else:
            raise AssertionError("invalid input accepted")
    try:
        weekly_target(monday + timedelta(days=1), *_window(monday))
    except ValueError:
        pass
    else:
        raise AssertionError("Tuesday decision accepted")


def _zip_month(*, header: bool = False, invalid_day: int | None = None,
               bad_volume: bool = False) -> bytes:
    rows = []
    if header:
        rows.append(list(HEADER))
    for i in range(30):
        day = date(2020, 11, 1) + timedelta(days=i)
        stamp = int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp()) * 1000
        if i == invalid_day:
            stamp += 1
        rows.append([stamp, 100, 101, 99, 100, 10, stamp + 86400000 - 1,
                     0 if bad_volume and i == 5 else 1000, 10, 5, 500, 0])
    body = "\n".join(",".join(map(str, row)) for row in rows).encode()
    out = BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("BTCUSDT-1d-2020-11.csv", body)
    return out.getvalue()


def test_futures_parser_accepts_optional_exact_header_and_rejects_bad_rows():
    for header in (False, True):
        values = _parse_futures_zip(_zip_month(header=header), "2020-11")
        assert len(values) == 30 and values[date(2020, 11, 1)] == 1000.0
    for body in (_zip_month(invalid_day=5), _zip_month(bad_volume=True)):
        try:
            _parse_futures_zip(body, "2020-11")
        except SourceFailure:
            pass
        else:
            raise AssertionError("invalid archive accepted")


def test_account_first_monday_terminal_sale_and_tuesday_stress():
    index = pd.date_range("2020-11-01", "2025-12-31", tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                           "close": 100.0, "quote_volume": 100_000_000.0}, index=index)
    futures = {stamp.date(): 100_000_000.0 * .999 ** i for i, stamp in enumerate(index)}
    result = simulate(futures, prices, 100.0)
    assert [x["day"] for x in result.fills] == ["2021-01-04", "2026-01-01"]
    assert result.terminal_cash < 10000.0
    delayed = simulate(futures, prices, 100.0, latency_days=1)
    assert delayed.fills[0]["day"] == "2021-01-05"
