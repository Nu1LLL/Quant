"""Synthetic-only causal, source-format, cash-account and Gate checks."""

from datetime import date, timedelta
from io import BytesIO
import json
import math
import zipfile

import pandas as pd

from research.btc_exchange_supply_sim import evaluate, simulate, weekly_target
from research.btc_exchange_supply_source import _parse_bn_zip, _terminal_open


def _synthetic():
    index = pd.date_range(date(2020, 12, 1), date(2025, 12, 31), tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100., "high": 100., "low": 100.,
                           "close": 100., "base_volume": 1e6,
                           "quote_volume": 1e9}, index=index)
    supply = {date(2020, 10, 1) + timedelta(days=i): 1000 - .01 * i
              for i in range(1918)}
    return supply, prices


def test_completed_sunday_and_28_day_direction_only():
    supply, _ = _synthetic()
    thu = date(2021, 1, 7)
    assert weekly_target(thu, supply) == 1
    supply[date(2021, 1, 4)] = 1e9
    supply[date(2021, 1, 6)] = 1e9
    assert weekly_target(thu, supply) == 1
    supply[date(2021, 1, 3)] = supply[date(2020, 12, 6)]
    assert weekly_target(thu, supply) == 0


def test_constant_price_fees_lot_capacity_terminal_and_stress():
    supply, prices = _synthetic()
    base = simulate(supply, prices, 100.)
    assert len(base.days) == 1826
    assert [(f["day"], f["side"]) for f in base.fills] == [
        ("2021-01-07", "buy"), ("2026-01-01", "sell")]
    buy, sell = base.fills
    assert math.isclose(buy["quantity"] / .00001,
                        round(buy["quantity"] / .00001), abs_tol=1e-7)
    assert math.isclose(base.terminal_cash,
                        10000 - buy["notional"] - buy["fee"] +
                        sell["notional"] - sell["fee"], rel_tol=1e-12)
    assert 0 < base.terminal_cash < 10000
    assert base.days[-1].end == base.terminal_cash
    assert simulate(supply, prices, 100., latency_days=1).fills[0]["day"] == "2021-01-08"
    assert simulate(supply, prices, 100., slippage=.001).terminal_cash < base.terminal_cash


def test_binance_timestamp_units_and_terminal_first_open():
    for month, unit in (("2020-12", 1000), ("2025-01", 1000000)):
        first = pd.Timestamp(month + "-01", tz="UTC")
        next_month = first + pd.offsets.MonthBegin(1)
        lines = []
        for day in pd.date_range(first, next_month - pd.Timedelta(days=1), freq="D"):
            stamp = int(day.timestamp()) * unit
            lines.append(f"{stamp},100,101,99,100,1,{stamp+86400*unit-1},100,1,1,1,0")
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr(f"BTCUSDT-1d-{month}.csv", "\n".join(lines))
        frame, audit = _parse_bn_zip(buf.getvalue(), month, False)
        assert len(frame) == len(lines)
        assert audit["timestamp_unit"] == ("us" if unit == 1000000 else "ms")
    stamp = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp()) * 1000000
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("BTCUSDT-1d-2026-01.csv",
                         f"{stamp},100,101,99,100,1,{stamp+86400*1000000-1},100,1,1,1,0\n"
                         "future rows intentionally ignored")
    assert _terminal_open(buf.getvalue()) == 100.


def test_full_gate_is_serializable_and_oos_cannot_pass():
    supply, prices = _synthetic()
    report = evaluate(simulate(supply, prices, 100.),
                      simulate(supply, prices, 100., slippage=0, fee_rate=0),
                      simulate(supply, prices, 100., slippage=.001),
                      simulate(supply, prices, 100., latency_days=1), prices)
    assert report["monte_carlo"]["paths"] == 5000
    assert len(report["annual"]) == 5
    assert len(report["monthly"]["returns"]) == 60
    assert report["checks"]["independent_executable_oos"] is False
    assert report["all_gate_passed"] is False
    json.dumps(report, allow_nan=False)
