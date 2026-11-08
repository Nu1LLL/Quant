"""Archive transport tests; synthetic trade IDs contain no market outcome."""

import pytest

from research.crypto_blackbox.deribit_history_archive import (
    ArchiveFailure, fetch_window,
)


class Response:
    def __init__(self, rows, more=False):
        self.payload = {"result": {"trades": rows, "has_more": more}}

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self, windows):
        self.windows = windows
        self.calls = []

    def get(self, _url, params, timeout):
        window = (params["start_timestamp"], params["end_timestamp"])
        self.calls.append(window)
        return self.windows[window]


def test_id_inversion_does_not_discard_valid_timestamped_trades():
    # The Deribit page observed on 2022-05-09 had this structural shape:
    # timestamps sorted, IDs unique, one numeric ID inversion.
    session = Session({(10, 13): Response([
        {"trade_id": "22", "timestamp": 10, "instrument_name": "A"},
        {"trade_id": "20", "timestamp": 11, "instrument_name": "B"},
    ])})
    result = fetch_window("BTC", "option", 10, 13, session=session)
    assert result.requests == 1
    assert len(result.rows) == 2
    assert result.current_vintage_only and not result.execution_eligible


def test_full_page_bisection_uses_disjoint_inclusive_intervals():
    session = Session({
        (10, 13): Response([], more=True),
        (10, 11): Response([{"trade_id": "1", "timestamp": 11, "instrument_name": "A"}]),
        (12, 13): Response([{"trade_id": "2", "timestamp": 12, "instrument_name": "B"}]),
    })
    result = fetch_window("BTC", "option", 10, 13, session=session)
    assert session.calls == [(10, 13), (10, 11), (12, 13)]
    assert [row["timestamp"] for row in result.rows] == [11, 12]
    assert result.requests == 3 and result.split_windows == 1


def test_duplicate_across_pages_and_overfull_millisecond_fail():
    repeated = Session({
        (10, 13): Response([], more=True),
        (10, 11): Response([{"trade_id": "1", "timestamp": 11,
                              "instrument_name": "A"}]),
        (12, 13): Response([{"trade_id": "1", "timestamp": 12,
                              "instrument_name": "A"}]),
    })
    with pytest.raises(ArchiveFailure, match="duplicate"):
        fetch_window("BTC", "option", 10, 13, session=repeated)
    overfull = Session({(10, 10): Response([], more=True)})
    with pytest.raises(ArchiveFailure, match="single-millisecond"):
        fetch_window("BTC", "option", 10, 10, session=overfull)


def test_out_of_window_and_invalid_interval_fail():
    session = Session({(10, 13): Response([{"trade_id": "1", "timestamp": 14,
                                           "instrument_name": "A"}])})
    with pytest.raises(ArchiveFailure, match="outside"):
        fetch_window("BTC", "option", 10, 13, session=session)
    with pytest.raises(ValueError, match="under one day"):
        fetch_window("BTC", "option", 10, 10 + 86_400_000, session=session)
