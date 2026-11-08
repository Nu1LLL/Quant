"""Synthetic checks for independent source timing and spot-account settlement."""

from datetime import date, timedelta

import pandas as pd
import pytest

from research.btc_exchange_supply_source import SourceFailure
from research import deribit_option_taker_source as source
from research.deribit_option_taker_sim import simulate


def _tape(monday: date, put_direction: str, call_direction: str) -> list[dict]:
    first, _ = source._monday_window(monday)
    rows = []
    for i in range(10):
        put = i < 5
        rows.append({"trade_id": str(i + 1), "timestamp": first + 1000 * i,
                     "instrument_name": ("BTC-25JUN21-30000-P" if put else
                                         "BTC-25JUN21-50000-C"),
                     "amount": 1, "index_price": 40000,
                     "direction": put_direction if put else call_direction})
    return rows


def test_option_taker_sign_and_exclusion(monkeypatch):
    monday = date(2021, 6, 14)
    rows = _tape(monday, "sell", "buy")
    monkeypatch.setattr(source, "_request", lambda *_: rows)
    _, target, audit = source._weekly(monday)
    assert target == 1 and audit["put_rows"] == audit["call_rows"] == 5
    rows = _tape(monday, "buy", "sell")
    monkeypatch.setattr(source, "_request", lambda *_: rows)
    assert source._weekly(monday)[1] == 0
    rows[0]["block_trade_id"] = "block"
    assert source._weekly(monday)[2]["low_density"] is True


def test_duplicate_trade_rejected(monkeypatch):
    monday = date(2021, 6, 14)
    rows = _tape(monday, "sell", "buy")
    rows[1]["trade_id"] = rows[0]["trade_id"]
    monkeypatch.setattr(source, "_request", lambda *_: rows)
    with pytest.raises(SourceFailure, match="uniqueness"):
        source._weekly(monday)


def test_overfull_page_is_split_without_skipping_millisecond(monkeypatch):
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Session:
        def get(self, _url, params, timeout):
            bounds = (params["start_timestamp"], params["end_timestamp"])
            calls.append(bounds)
            payload = ({"trades": [], "has_more": True} if bounds == (10, 13)
                       else {"trades": [{"timestamp": bounds[0]}], "has_more": False})
            return Response({"result": payload})

    monkeypatch.setattr(source, "_session", lambda: Session())
    rows = source._request(10, 13)
    assert calls == [(10, 13), (10, 11), (12, 13)]
    assert [row["timestamp"] for row in rows] == [10, 12]


def test_account_first_decision_and_terminal_fee():
    idx = pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                           "close": 100.0, "quote_volume": 1_000_000.0}, index=idx)
    targets = {date(2021, 1, 4) + timedelta(days=7 * i): 1 for i in range(261)}
    replay = simulate(targets, prices, 100.0)
    assert replay.fills[0]["day"] == "2021-01-05"
    assert replay.fills[-1]["day"] == "2026-01-01"
    assert replay.terminal_cash < 10_000
    assert replay.days[-1].end == replay.terminal_cash
    late = simulate(targets, prices, 100.0, latency_days=1)
    assert late.fills[0]["day"] == "2021-01-06"
    targets[date(2021, 1, 11)] = 0
    exit_replay = simulate(targets, prices, 100.0)
    assert exit_replay.fills[1]["day"] == "2021-01-12"
    assert exit_replay.fills[1]["side"] == "sell"
