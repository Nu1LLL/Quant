"""Synthetic causal regression, delayed release and account checks."""

from datetime import date, timedelta
from hashlib import sha256
from io import BytesIO, StringIO
import csv
import math
import zipfile

import pandas as pd

from research.cme_option_pressure_source import OptionReport, _parse_zip
from research.cme_option_pressure_sim import rolling_residual, simulate, weekly_target


def _flat_prices():
    idx = pd.date_range("2020-06-01", "2025-12-31", tz="UTC", freq="D")
    return pd.DataFrame({"open": 100., "high": 100., "low": 100.,
                         "close": 100., "quote_volume": 1e9}, index=idx)


def _reports():
    return tuple(OptionReport(date(2020, 1, 7) + timedelta(days=7 * i),
                              0.0 if i == 51 else .1) for i in range(313))


def test_residual_uses_only_prior_twenty_six_reports():
    reports = _reports()
    closes = {ts.date(): 100. for ts in _flat_prices().index}
    assert rolling_residual(51, reports, closes) < 0
    assert weekly_target(date(2021, 1, 6), reports, closes) == 1
    assert weekly_target(date(2021, 1, 13), reports, closes) == 0
    later = list(reports)
    later[70] = OptionReport(later[70].report_date, -.8)
    assert weekly_target(date(2021, 1, 6), tuple(later), closes) == 1


def test_flat_account_forced_close_and_costs():
    prices = _flat_prices()
    result = simulate(_reports(), prices, 100.)
    assert len(result.days) == 1826
    assert result.fills[0]["day"] == "2021-01-06"
    assert result.fills[0]["side"] == "buy"
    assert result.fills[1]["day"] == "2021-01-13"
    assert result.fills[1]["side"] == "sell"
    assert math.isclose(result.days[-1].end, result.terminal_cash, abs_tol=1e-8)
    assert result.terminal_cash < 10000
    hold = simulate(_reports(), prices, 100., buy_and_hold=True)
    assert hold.fills[0]["day"] == "2021-01-01"
    assert hold.fills[-1]["day"] == "2026-01-01"


def test_cftc_option_archive_member_and_hash():
    fields = ["Market_and_Exchange_Names", "Report_Date_as_YYYY-MM-DD",
              "Open_Interest_All", "Asset_Mgr_Positions_Long_All",
              "Asset_Mgr_Positions_Short_All"] + [f"other_{i}" for i in range(82)]
    raw = StringIO()
    writer = csv.DictWriter(raw, fieldnames=fields)
    writer.writeheader()
    for i in range(52):
        writer.writerow({"Market_and_Exchange_Names": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
                         "Report_Date_as_YYYY-MM-DD": str(date(2020, 1, 7) + timedelta(days=7 * i)),
                         "Open_Interest_All": "100", "Asset_Mgr_Positions_Long_All": "30",
                         "Asset_Mgr_Positions_Short_All": "10"})
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("FinComYY.txt", raw.getvalue())
    body = buf.getvalue()
    rows = _parse_zip(body, 2020, "com", sha256(body).hexdigest())
    assert len(rows) == 52
    assert rows[date(2020, 1, 7)] == (100., 30., 10.)
    try:
        _parse_zip(body, 2020, "com", "0" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("changed CFTC combined archive accepted")
