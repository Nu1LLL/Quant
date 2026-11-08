"""Synthetic source, Sunday-decision and ETH account checks only."""

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import json
import zipfile

import pandas as pd

from research import eth_net_supply_source as source
from research.eth_net_supply_sim import simulate, weekly_target


def _supply() -> dict[date, float]:
    return {date(2021, 7, 1) + timedelta(days=i): float(1_000_000 - i)
            for i in range(1918)}


def test_thursday_uses_prior_sunday_and_28_day_change():
    levels = _supply()
    thursday = date(2021, 10, 7)
    assert weekly_target(thursday, levels) == 1
    levels[date(2021, 10, 4)] = 2_000_000
    assert weekly_target(thursday, levels) == 1
    levels[date(2021, 10, 3)] = levels[date(2021, 9, 5)]
    assert weekly_target(thursday, levels) == 0
    del levels[date(2021, 9, 5)]
    try:
        weekly_target(thursday, levels)
    except source.SourceFailure:
        pass
    else:
        raise AssertionError("missing older supply day accepted")


def test_unpinned_supply_response_is_rejected_before_numeric_parse():
    try:
        source.parse_supply(b'{"data":[{"asset":"eth","SplyCur":"1"}]}')
    except source.SourceFailure as exc:
        assert "response changed" in str(exc)
    else:
        raise AssertionError("unpinned supply response accepted")


def test_complete_synthetic_supply_grid():
    rows = [{"asset": "eth", "time": str(day) + "T00:00:00.000000000Z",
             "SplyCur": str(int(value))} for day, value in _supply().items()]
    body = json.dumps({"data": rows, "next_page_url": None}, separators=(",", ":")).encode()
    pin = source.SUPPLY_SHA
    source.SUPPLY_SHA = sha256(body).hexdigest()
    try:
        assert len(source.parse_supply(body)) == 1918
    finally:
        source.SUPPLY_SHA = pin


def test_terminal_open_reads_only_first_csv_row():
    expected = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
    data = BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("ETHUSDT-1d-2026-10.csv",
                         ",".join([str(expected), "100"] + ["0"] * 10) + "\n"
                         + "not,valid,a,later,row\n")
    assert source._terminal_open(data.getvalue()) == 100.0


def test_account_starts_on_first_thursday_and_liquidates_terminally():
    idx = pd.date_range("2021-08-01", "2026-09-30", tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                           "close": 100.0, "volume": 100_000_000.0}, index=idx)
    result = simulate(_supply(), prices, 100.0)
    assert [x["day"] for x in result.fills] == ["2021-10-07", "2026-10-01"]
    assert result.terminal_cash < 10_000
    delayed = simulate(_supply(), prices, 100.0, latency_days=1)
    assert delayed.fills[0]["day"] == "2021-10-08"
