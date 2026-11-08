"""Causal lag and account reconciliation checks for the frozen CME hypothesis."""

from datetime import date, timedelta
from hashlib import sha256
from io import BytesIO, StringIO
import csv
import math
import zipfile

import pandas as pd

from research.cme_asset_manager_source import Position, parse_archive
from research.cme_asset_manager_sim import simulate, weekly_target


def test_position_signal_waits_thirteen_days_and_four_reports():
    dates = [date(2020, 11, 24) + timedelta(days=7 * i) for i in range(6)]
    reports = tuple(Position(day, ratio) for day, ratio in zip(
        dates, (.05, .10, .15, .20, .30, -.40)))
    assert weekly_target(date(2021, 1, 4), reports) == 1
    assert weekly_target(date(2021, 1, 11), reports) == 0
    try:
        weekly_target(date(2021, 1, 5), reports)
    except ValueError:
        pass
    else:
        raise AssertionError("non-Monday decision accepted")


def test_flat_market_costs_and_forced_terminal_sale_reconcile():
    idx = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100., "high": 100., "low": 100.,
                           "close": 100., "quote_volume": 1e9}, index=idx)
    reports = tuple(Position(date(2020, 1, 7) + timedelta(days=7 * i),
                             .01 + i / 1000) for i in range(313))
    result = simulate(reports, prices, 100.)
    assert len(result.days) == 1826
    assert [x["side"] for x in result.fills] == ["buy", "sell"]
    assert result.fills[0]["day"] == "2021-01-04"
    assert result.fills[-1]["day"] == "2026-01-01"
    assert math.isclose(result.terminal_cash, result.days[-1].end, abs_tol=1e-8)
    assert result.terminal_cash < 10000
    assert result.fees > 0
    hold = simulate(reports, prices, 100., buy_and_hold=True)
    assert hold.fills[0]["day"] == "2021-01-01"
    assert math.isclose(hold.terminal_cash, hold.days[-1].end, abs_tol=1e-8)


def test_cftc_archive_selects_standard_contract_and_rejects_changed_hash():
    fields = ["Market_and_Exchange_Names", "Report_Date_as_YYYY-MM-DD",
              "Open_Interest_All", "Asset_Mgr_Positions_Long_All",
              "Asset_Mgr_Positions_Short_All"] + [f"other_{i}" for i in range(82)]
    raw = StringIO()
    writer = csv.DictWriter(raw, fieldnames=fields)
    writer.writeheader()
    for i in range(52):
        writer.writerow({"Market_and_Exchange_Names": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
                         "Report_Date_as_YYYY-MM-DD": str(date(2020, 1, 7) + timedelta(days=i * 7)),
                         "Open_Interest_All": "100", "Asset_Mgr_Positions_Long_All": "30",
                         "Asset_Mgr_Positions_Short_All": "10"})
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("FinFutYY.txt", raw.getvalue())
    body = buf.getvalue()
    positions = parse_archive(body, 2020, sha256(body).hexdigest())
    assert len(positions) == 52
    assert math.isclose(positions[0].net_oi_ratio, .2)
    try:
        parse_archive(body, 2020, "0" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("altered CFTC archive hash accepted")
