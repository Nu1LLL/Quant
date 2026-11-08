"""Synthetic causal, source-format, cost and full-report checks."""

from datetime import date, timedelta
from io import BytesIO
import json
import math
import zipfile

import pandas as pd

from research.coinbase_premium_sim import evaluate, premium_series, simulate, weekly_target
from research.coinbase_premium_source import _parse_bn_zip


def _synthetic():
    bn_index = pd.date_range(date(2021, 8, 1), date(2026, 9, 30), tz="UTC", freq="D")
    cb_index = pd.date_range(date(2021, 9, 1), date(2026, 9, 30), tz="UTC", freq="D")
    bn = pd.DataFrame({"open": 10000., "high": 10000., "low": 10000.,
                       "close": 10000., "base_volume": 100.,
                       "quote_volume": 1e9}, index=bn_index)
    cb_btc = pd.DataFrame({"close": 10010.}, index=cb_index)
    cb_usdt = pd.DataFrame({"close": 1.}, index=cb_index)
    return cb_btc, cb_usdt, bn


def test_usdt_usd_conversion_changes_premium_sign_without_future_data():
    cb_btc, cb_usdt, bn = _synthetic()
    p = premium_series(cb_btc, cb_usdt, bn)
    assert weekly_target(date(2021, 10, 5), p) == 1
    cb_usdt.loc[pd.Timestamp("2021-10-05", tz="UTC"), "close"] = 100.
    assert weekly_target(date(2021, 10, 5), premium_series(cb_btc, cb_usdt, bn)) == 1
    cb_usdt.loc[pd.Timestamp("2021-10-03", tz="UTC"), "close"] = 2.
    assert weekly_target(date(2021, 10, 5), premium_series(cb_btc, cb_usdt, bn)) == 0


def test_constant_price_account_charges_fees_rounds_lots_and_terminal():
    cb_btc, cb_usdt, bn = _synthetic()
    p = premium_series(cb_btc, cb_usdt, bn)
    baseline = simulate(p, bn, 10000.)
    assert len(baseline.days) == 1826
    assert [(x["day"], x["side"]) for x in baseline.fills] == [
        ("2021-10-05", "buy"), ("2026-10-01", "sell")]
    buy, sell = baseline.fills
    assert math.isclose(buy["quantity"] / .00001, round(buy["quantity"] / .00001), abs_tol=1e-7)
    expected = 10000. - buy["notional"] - buy["fee"] + sell["notional"] - sell["fee"]
    assert math.isclose(baseline.terminal_cash, expected, rel_tol=1e-12)
    assert baseline.terminal_cash < 10000.
    assert math.isclose(baseline.days[-1].end, baseline.terminal_cash, rel_tol=1e-12)


def test_timestamp_unit_parse_and_full_metrics_serialize():
    for month, unit in (("2021-08", 1000), ("2025-01", 1000000)):
        first = pd.Timestamp(month + "-01", tz="UTC")
        nxt = first + pd.offsets.MonthBegin(1)
        lines = []
        for d in pd.date_range(first, nxt - pd.Timedelta(days=1), freq="D"):
            t = int(d.timestamp()) * unit
            lines.append(f"{t},100,101,99,100,1,{t+86400*unit-1},100,1,1,1,0")
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(f"BTCUSDT-1d-{month}.csv", "\n".join(lines))
        frame, audit = _parse_bn_zip(buf.getvalue(), month, False)
        assert len(frame) == len(lines)
        assert audit["timestamp_unit"] == ("us" if unit == 1000000 else "ms")
    cb_btc, cb_usdt, bn = _synthetic()
    p = premium_series(cb_btc, cb_usdt, bn)
    report = evaluate(simulate(p, bn, 10000.),
                      simulate(p, bn, 10000., slippage=0, fee_rate=0),
                      simulate(p, bn, 10000., slippage=.001),
                      simulate(p, bn, 10000., latency_days=1), bn)
    assert report["monte_carlo"]["paths"] == 5000
    assert len(report["annual"]) == 6
    assert len(report["monthly"]["returns"]) == 60
    json.dumps(report, allow_nan=False)
