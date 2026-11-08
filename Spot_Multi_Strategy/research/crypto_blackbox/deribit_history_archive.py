"""Bounded, read-only Deribit history transport for *future* preregistrations.

This module has no signal, return, backtest or execution path. It does not
repair or rerun any sealed option-pressure experiment. Deribit's public
history has been observed with a numeric trade-ID inversion inside a
timestamp-ordered page, so ID monotonicity is not a source requirement here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import time
from typing import Callable

import requests


ENDPOINT = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
MAX_ROWS = 1000
MAX_WINDOW_MS = 86_400_000
_CURRENCIES = frozenset({"BTC", "ETH"})
_KINDS = frozenset({"option", "future"})


class ArchiveFailure(ValueError):
    """A source-integrity, API, pagination or boundedness failure."""


@dataclass(frozen=True)
class ArchiveWindow:
    currency: str
    kind: str
    start_ms: int
    end_ms: int
    rows: tuple[dict, ...]
    canonical_sha256: str
    requests: int
    split_windows: int
    current_vintage_only: bool = True
    execution_eligible: bool = False


def _request_page(session: requests.Session, params: dict,
                  sleep: Callable[[float], None]) -> tuple[list[dict], bool]:
    last: Exception | None = None
    for attempt, delay in enumerate((0, 1, 2, 4)):
        if delay:
            sleep(delay)
        try:
            response = session.get(ENDPOINT, params=params, timeout=25)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("error") is not None:
                raise ArchiveFailure("Deribit history returned an API error")
            result = payload.get("result")
            if not isinstance(result, dict):
                raise ArchiveFailure("Deribit history result is not an object")
            rows = result.get("trades")
            more = result.get("has_more")
            if not isinstance(rows, list) or type(more) is not bool or len(rows) > MAX_ROWS:
                raise ArchiveFailure("Deribit history page schema/limit changed")
            return rows, more
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt == 3:
                break
    raise ArchiveFailure(f"Deribit history request exhausted retries: {type(last).__name__}")


def fetch_window(currency: str, kind: str, start_ms: int, end_ms: int,
                 *, session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> ArchiveWindow:
    """Fetch one inclusive UTC-ms interval, splitting every truncated page.

    The caller chooses windows without inspecting return outcomes. A successful
    response proves only that this *current-vintage* endpoint returned a
    complete page under its own has_more contract, not historical first-seen
    availability or executable option quotes.
    """
    if currency not in _CURRENCIES or kind not in _KINDS:
        raise ValueError("unsupported frozen currency or instrument kind")
    if (type(start_ms) is not int or type(end_ms) is not int
            or start_ms <= 0 or end_ms < start_ms
            or end_ms - start_ms >= MAX_WINDOW_MS):
        raise ValueError("window must be a positive inclusive interval under one day")
    own_session = session is None
    active = session if session is not None else requests.Session()
    rows: list[dict] = []
    calls = splits = 0
    try:
        def collect(first: int, last: int) -> None:
            nonlocal calls, splits
            params = {"currency": currency, "kind": kind, "include_old": "true",
                      "count": MAX_ROWS, "sorting": "asc",
                      "start_timestamp": first, "end_timestamp": last}
            page, more = _request_page(active, params, sleep)
            calls += 1
            if more:
                if first == last:
                    raise ArchiveFailure("overfull single-millisecond archive window")
                splits += 1
                middle = (first + last) // 2
                collect(first, middle)
                collect(middle + 1, last)
                return
            for row in page:
                if not isinstance(row, dict):
                    raise ArchiveFailure("non-object trade row")
                ident, stamp, name = (row.get("trade_id"), row.get("timestamp"),
                                      row.get("instrument_name"))
                if (not isinstance(ident, str) or not ident or type(stamp) is not int
                        or not isinstance(name, str) or not name):
                    raise ArchiveFailure("trade identity or timestamp missing")
                if not first <= stamp <= last:
                    raise ArchiveFailure("trade timestamp outside requested interval")
                rows.append(row)

        collect(start_ms, end_ms)
    finally:
        if own_session:
            active.close()
    ids = [row["trade_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ArchiveFailure("duplicate currency-level trade ID across archive pages")
    rows.sort(key=lambda row: (row["timestamp"], row["trade_id"],
                               str(row.get("instrument_name", ""))))
    try:
        body = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ArchiveFailure("non-canonical archive response") from exc
    return ArchiveWindow(currency, kind, start_ms, end_ms, tuple(rows),
                         sha256(body).hexdigest(), calls, splits)
