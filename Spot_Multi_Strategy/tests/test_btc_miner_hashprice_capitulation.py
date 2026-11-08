"""Synthetic-only miner stress timing, source, account and report checks."""

from datetime import date, timedelta
from hashlib import sha256
import json
import math

import pandas as pd

import research.btc_miner_hashprice_source as source
from research.btc_miner_hashprice_sim import evaluate, simulate, weekly_target


def _synthetic():
    metric_dates = pd.date_range(date(2020, 1, 1), date(2025, 12, 31), tz="UTC", freq="D")
    metrics = pd.DataFrame({"IssTotNtv": 6., "FeeTotNtv": 0.,
                            "HashRate": [1000 - .01 * i for i in range(len(metric_dates))],
                            "PriceUSD": [100 * math.exp(-.001 * i)
                                         for i in range(len(metric_dates))]},
                           index=metric_dates)
    price_dates = pd.date_range(date(2020, 12, 1), date(2025, 12, 31), tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100., "high": 100., "low": 100.,
                           "close": 100., "base_volume": 1e6,
                           "quote_volume": 1e9}, index=price_dates)
    return metrics, prices


def test_sunday_publication_buffer_and_joint_stress_conditions():
    metrics, _ = _synthetic()
    thu = date(2021, 1, 7)
    assert weekly_target(thu, metrics) == 1
    metrics.loc[pd.Timestamp("2021-01-04", tz="UTC"), "PriceUSD"] = 1e12
    metrics.loc[pd.Timestamp("2021-01-06", tz="UTC"), "PriceUSD"] = 1e12
    assert weekly_target(thu, metrics) == 1
    metrics.loc[pd.Timestamp("2021-01-03", tz="UTC"), "PriceUSD"] = 1e12
    assert weekly_target(thu, metrics) == 0
    metrics, _ = _synthetic()
    recent = pd.date_range("2020-12-07", "2021-01-03", tz="UTC", freq="D")
    metrics.loc[recent, "HashRate"] = 2000.
    assert weekly_target(thu, metrics) == 0


def test_costed_constant_price_account_and_friday_stress():
    metrics, prices = _synthetic()
    base = simulate(metrics, prices, 100.)
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
    assert simulate(metrics, prices, 100., latency_days=1).fills[0]["day"] == "2021-01-08"
    assert simulate(metrics, prices, 100., slippage=.001).terminal_cash < base.terminal_cash


def test_frozen_payload_requires_exact_dates_and_schema():
    rows = []
    for day in pd.date_range("2020-01-01", "2025-12-31", tz="UTC", freq="D"):
        rows.append({"asset": "btc", "time": day.isoformat(), "IssTotNtv": "6",
                     "FeeTotNtv": "1", "HashRate": "100", "PriceUSD": "100"})
    body = json.dumps({"data": rows, "next_page_url": None}).encode()
    original = source.CM_SHA
    try:
        source.CM_SHA = sha256(body).hexdigest()
        parsed = source.parse_coinmetrics(body)
        assert len(parsed) == 2192
        assert parsed.iloc[0]["HashRate"] == 100
        rows[1]["time"] = rows[0]["time"]
        changed = json.dumps({"data": rows, "next_page_url": None}).encode()
        source.CM_SHA = sha256(changed).hexdigest()
        try:
            source.parse_coinmetrics(changed)
        except source.SourceFailure:
            pass
        else:
            raise AssertionError("duplicate UTC day accepted")
    finally:
        source.CM_SHA = original


def test_full_report_serializes_and_oos_is_rejected():
    metrics, prices = _synthetic()
    report = evaluate(simulate(metrics, prices, 100.),
                      simulate(metrics, prices, 100., slippage=0, fee_rate=0),
                      simulate(metrics, prices, 100., slippage=.001),
                      simulate(metrics, prices, 100., latency_days=1), prices)
    assert report["monte_carlo"]["paths"] == 5000
    assert len(report["annual"]) == 5
    assert report["monthly"]["months"] == 60
    assert report["checks"]["independent_executable_oos"] is False
    assert report["all_gate_passed"] is False
    json.dumps(report, allow_nan=False)
