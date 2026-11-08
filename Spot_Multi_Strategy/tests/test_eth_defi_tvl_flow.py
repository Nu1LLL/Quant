"""Causal and accounting checks on synthetic values only."""

from datetime import date
from io import BytesIO
import math
import zipfile

import pandas as pd

from research.eth_defi_tvl_sim import _target, simulate
from research.eth_defi_tvl_source import _parse_spot_zip


def test_target_uses_only_prior_completed_sundays():
    closes = {date(2020, 12, 6): 100., date(2021, 1, 3): 100.}
    tvl = {date(2020, 12, 6): 1000., date(2021, 1, 3): 1100.,
           date(2021, 1, 4): 1.}
    assert _target(date(2021, 1, 7), tvl, closes) == 1
    tvl[date(2021, 1, 4)] = 1e20
    assert _target(date(2021, 1, 7), tvl, closes) == 1
    tvl[date(2021, 1, 3)] = 900.
    assert _target(date(2021, 1, 7), tvl, closes) == 0


def test_constant_price_one_roundtrip_charges_both_fills_and_terminal():
    index = pd.date_range(date(2020, 11, 1), date(2025, 12, 31), tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100., "high": 100., "low": 100.,
                           "close": 100., "volume": 10000.}, index=index)
    tvl = {ts.date(): 1000. + i for i, ts in enumerate(index)}
    result = simulate(tvl, prices, 100.)
    assert len(result.days) == 1826
    assert [(t["day"], t["side"]) for t in result.trades] == [
        ("2021-01-07", "buy"), ("2026-01-01", "sell")]
    expected = 10000. * (1 - .0005) * (1 - .001) / ((1 + .0005) * (1 + .001))
    assert math.isclose(result.terminal_cash, expected, rel_tol=1e-12)
    assert math.isclose(result.days[-1].end, result.terminal_cash, rel_tol=1e-12)
    assert math.isclose(result.days[-1].fee, result.trades[-1]["fee"], rel_tol=1e-12)


def test_spot_archive_parses_both_documented_timestamp_units():
    for month, unit in (("2020-11", 1000), ("2025-01", 1000000)):
        first = pd.Timestamp(month + "-01", tz="UTC")
        following = first + pd.offsets.MonthBegin(1)
        lines = []
        for day in pd.date_range(first, following - pd.Timedelta(days=1), freq="D"):
            stamp = int(day.timestamp()) * unit
            lines.append(f"{stamp},100,101,99,100,1,{stamp + 86400*unit - 1},100,1,1,1,0")
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr(f"ETHUSDT-1d-{month}.csv", "\n".join(lines))
        frame, audit = _parse_spot_zip(buf.getvalue(), month, False)
        assert len(frame) == len(lines)
        assert audit["timestamp_unit"] == ("us" if unit == 1000000 else "ms")
