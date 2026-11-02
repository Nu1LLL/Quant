"""Fail-closed Binance archive loader for the fixed Fed-reserves experiment.

Only official USD-M monthly 4h trade, 4h mark, and funding ZIPs are read.
An entire missing UTC day in a monthly bar archive can be replaced solely by
the corresponding official daily 4h archive. Each ZIP is SHA-256 checked
against its own `.CHECKSUM`; raw payloads remain in memory. This module does
not fetch macro observations, calculate signals, or compute returns.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from typing import Callable

import pandas as pd

from research.altcoin_reversal_archive import _month_range
from research.wiki_attention_data import (
    ArchiveIntegrityError,
    _Payload,
    _audit_bars,
    _audit_funding,
    _http_fetch,
    _load_archive,
)
from research.wiki_attention_daily_archive_data import (
    _full_missing_days,
    _load_daily_archive,
)


SYMBOLS = ("BTCUSDT", "ETHUSDT", "DOGEUSDT", "LTCUSDT", "BCHUSDT")
START = pd.Timestamp("2020-12-01", tz="UTC")
TERMINAL = pd.Timestamp("2026-01-01", tz="UTC")
LAST_ARCHIVE_MONTH = "2025-12"
BAR_KINDS = ("klines", "markPriceKlines")
ARCHIVE_KINDS = BAR_KINDS + ("fundingRate",)
MONTHLY_BASE = "https://data.binance.vision/data/futures/um/monthly"
DAILY_BASE = "https://data.binance.vision/data/futures/um/daily"


@dataclass(frozen=True)
class DataBundle:
    trade_bars: dict[str, pd.DataFrame]
    mark_bars: dict[str, pd.DataFrame]
    funding_events: dict[str, pd.DataFrame]
    manifest: dict


def _load_and_audit_impl(
    *, symbols: tuple[str, ...], start: pd.Timestamp,
    terminal: pd.Timestamp, last_archive_month: str,
    fetcher: Callable[[str], bytes | _Payload], max_workers: int,
) -> DataBundle:
    """Private boundary injection for deterministic offline integrity tests."""
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be nonempty and unique")
    if max_workers < 1 or max_workers > 12:
        raise ValueError("max_workers must be 1..12")
    if start.tzinfo is None or terminal.tzinfo is None:
        raise ValueError("boundaries must be UTC-aware")
    if start.utcoffset() != pd.Timedelta(0) or terminal.utcoffset() != pd.Timedelta(0):
        raise ValueError("boundaries must be UTC")
    if start != pd.Timestamp(f"{start:%Y-%m}-01", tz="UTC"):
        raise ValueError("start must be a month start")
    if terminal != pd.Timestamp(f"{terminal:%Y-%m}-01", tz="UTC"):
        raise ValueError("terminal must be a month start")
    if start >= terminal:
        raise ValueError("start must precede terminal")
    months = _month_range(f"{start:%Y-%m}", last_archive_month)
    if months[-1] != f"{terminal - pd.Timedelta(days=1):%Y-%m}":
        raise ValueError("last_archive_month must immediately precede terminal")

    tasks = [
        (kind, symbol, month, False)
        for symbol in symbols for kind in ARCHIVE_KINDS for month in months
    ] + [
        (kind, symbol, f"{terminal:%Y-%m}", True)
        for symbol in symbols for kind in ARCHIVE_KINDS
    ]
    monthly: dict[tuple[str, str, str], pd.DataFrame] = {}
    archive_manifest: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _load_archive, kind, symbol, month, is_terminal,
                fetcher, terminal,
            ): (kind, symbol, month)
            for kind, symbol, month, is_terminal in tasks
        }
        for future in as_completed(futures):
            kind, symbol, month = futures[future]
            try:
                frame, detail = future.result()
            except IndexError as exc:
                for other in futures:
                    other.cancel()
                raise ArchiveIntegrityError(
                    f"{kind}/{symbol}/{month}: empty or malformed monthly CSV"
                ) from exc
            except Exception:
                for other in futures:
                    other.cancel()
                raise
            monthly[(kind, symbol, month)] = frame
            archive_manifest[f"monthly/{kind}/{symbol}/{month}"] = detail

    # A partial-day gap is never eligible for repair, and must fail before
    # any daily source is requested. Daily repair is forbidden for funding.
    missing: dict[tuple[str, str, str], tuple[pd.Timestamp, ...]] = {}
    for symbol in symbols:
        for kind in BAR_KINDS:
            for month in months:
                key = (kind, symbol, month)
                days = _full_missing_days(
                    monthly[key], kind=kind, symbol=symbol, month=month
                )
                missing[key] = days
                archive_manifest[f"monthly/{kind}/{symbol}/{month}"][
                    "missing_full_utc_days"
                ] = [f"{day:%Y-%m-%d}" for day in days]

    daily_tasks = [
        (kind, symbol, month, day)
        for symbol in symbols for kind in BAR_KINDS for month in months
        for day in missing[(kind, symbol, month)]
    ]
    daily: dict[tuple[str, str, str, pd.Timestamp], pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_load_daily_archive, kind, symbol, day, fetcher):
            (kind, symbol, month, day)
            for kind, symbol, month, day in daily_tasks
        }
        for future in as_completed(futures):
            kind, symbol, month, day = futures[future]
            try:
                frame, detail = future.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
            if not frame.index.intersection(monthly[(kind, symbol, month)].index).empty:
                raise ArchiveIntegrityError(
                    f"{kind}/{symbol}/{day.date()}: daily/monthly bar overlap"
                )
            daily[(kind, symbol, month, day)] = frame
            archive_manifest[f"daily/{kind}/{symbol}/{day:%Y-%m-%d}"] = detail

    segments: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    for (kind, symbol, _month), frame in monthly.items():
        segments[(kind, symbol)].append(frame)
    for (kind, symbol, _month, _day), frame in daily.items():
        segments[(kind, symbol)].append(frame)

    trade_bars: dict[str, pd.DataFrame] = {}
    mark_bars: dict[str, pd.DataFrame] = {}
    funding_events: dict[str, pd.DataFrame] = {}
    coverage: dict[str, dict] = {}
    for symbol in symbols:
        trade_bars[symbol], trade_detail = _audit_bars(
            segments[("klines", symbol)], kind="klines", symbol=symbol,
            start=start, terminal=terminal,
        )
        mark_bars[symbol], mark_detail = _audit_bars(
            segments[("markPriceKlines", symbol)], kind="markPriceKlines",
            symbol=symbol, start=start, terminal=terminal,
        )
        if not trade_bars[symbol].index.equals(mark_bars[symbol].index):
            raise ArchiveIntegrityError(f"{symbol}: trade/mark grids differ")
        funding_events[symbol], funding_detail = _audit_funding(
            segments[("fundingRate", symbol)], symbol=symbol,
            start=start, terminal=terminal,
            mark_index=mark_bars[symbol].index,
        )
        coverage[symbol] = {
            "trade": trade_detail, "mark": mark_detail,
            "funding": funding_detail,
        }

    replacements = [
        {"kind": kind, "symbol": symbol, "month": month,
         "date": f"{day:%Y-%m-%d}"}
        for kind, symbol, month, day in sorted(daily_tasks)
    ]
    manifest = {
        "registered_symbols": list(symbols),
        "monthly_source": MONTHLY_BASE,
        "daily_source": DAILY_BASE,
        "start": start.isoformat(),
        "terminal": terminal.isoformat(),
        "monthly_archive_count": len(tasks),
        "daily_archive_count": len(daily_tasks),
        "archive_count": len(archive_manifest),
        "daily_replacements": replacements,
        "archives": dict(sorted(archive_manifest.items())),
        "coverage": coverage,
    }
    json.dumps(manifest, allow_nan=False)
    return DataBundle(trade_bars, mark_bars, funding_events, manifest)


def load_and_audit(
    *, fetcher: Callable[[str], bytes | _Payload] | None = None,
    max_workers: int = 8,
) -> DataBundle:
    """Load exactly the preregistered five contracts and fixed historical grid.

    ``fetcher`` exists only for fixture injection; production boundaries and
    symbols cannot be changed through this public interface.
    """
    return _load_and_audit_impl(
        symbols=SYMBOLS, start=START, terminal=TERMINAL,
        last_archive_month=LAST_ARCHIVE_MONTH,
        fetcher=fetcher or _http_fetch, max_workers=max_workers,
    )
