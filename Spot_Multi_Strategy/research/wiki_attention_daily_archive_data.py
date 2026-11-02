"""Fail-closed official daily-archive repair for the fixed Wikipedia experiment.

This is a new data version, not a modification of the sealed monthly-only
experiment. A monthly 4h trade or mark archive may be supplemented only when
it lacks all six bars of an entire UTC day. Only that day's official 4h ZIP
and CHECKSUM are fetched. Raw ZIPs, CSVs, and pageviews stay in memory.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import re
from typing import Callable

import pandas as pd

from research.altcoin_reversal_archive import _month_range, _read_zip_csv
from research.wiki_attention_data import (
    ArchiveIntegrityError,
    BAR_PERIOD,
    DataBundle,
    LAST_ARCHIVE_MONTH,
    PAGE_END,
    PAGE_TITLES,
    START,
    TERMINAL,
    WIKIMEDIA_BASE,
    _Payload,
    _audit_bars,
    _audit_funding,
    _csv_header,
    _fetch,
    _http_fetch,
    _load_archive,
    _load_page,
    _parse_complete_bars,
)


DAILY_BASE = "https://data.binance.vision/data/futures/um/daily"
BAR_KINDS = ("klines", "markPriceKlines")
ARCHIVE_KINDS = BAR_KINDS + ("fundingRate",)


def _daily_archive_url(kind: str, symbol: str, day: pd.Timestamp) -> tuple[str, str]:
    if kind not in BAR_KINDS:
        raise ValueError(f"daily repair prohibited for {kind}")
    if not re.fullmatch(r"[A-Z0-9]+USDT", symbol):
        raise ValueError(f"invalid USD-M USDT symbol: {symbol}")
    if day.tzinfo is None or day.utcoffset() != pd.Timedelta(0) or day != day.normalize():
        raise ValueError("daily repair date must be UTC midnight")
    basename = f"{symbol}-4h-{day:%Y-%m-%d}.zip"
    return f"{DAILY_BASE}/{kind}/{symbol}/4h/{basename}", basename


def _full_missing_days(
    monthly: pd.DataFrame, *, kind: str, symbol: str, month: str
) -> tuple[pd.Timestamp, ...]:
    """Reject every partial-day or off-month hole before any daily fetch."""
    label = f"{kind}/{symbol}/{month}"
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    expected = pd.date_range(
        month_start, month_end - BAR_PERIOD, freq=BAR_PERIOD,
        tz="UTC", name="open_time",
    )
    if monthly.index.has_duplicates:
        raise ArchiveIntegrityError(f"{label}: duplicate monthly timestamps")
    outside = monthly.index.difference(expected)
    if len(outside):
        raise ArchiveIntegrityError(f"{label}: bars outside monthly 4h grid")
    missing = expected.difference(monthly.index)
    days = tuple(sorted(set(missing.normalize())))
    for day in days:
        day_grid = pd.date_range(
            day, day + pd.Timedelta(days=1) - BAR_PERIOD,
            freq=BAR_PERIOD, tz="UTC", name="open_time",
        )
        if not missing.intersection(day_grid).equals(day_grid):
            raise ArchiveIntegrityError(
                f"{label}: partial UTC day missing at {day.date()}"
            )
    return days


def _load_daily_archive(
    kind: str, symbol: str, day: pd.Timestamp,
    fetcher: Callable[[str], bytes | _Payload],
) -> tuple[pd.DataFrame, dict]:
    """Verify exactly one preregistered missing day's original-format bars."""
    url, basename = _daily_archive_url(kind, symbol, day)
    checksum_response = _fetch(fetcher, url + ".CHECKSUM")
    try:
        checksum = checksum_response.body.decode("ascii").strip()
    except UnicodeError as exc:
        raise ArchiveIntegrityError(f"{basename}: invalid checksum encoding") from exc
    matched = re.fullmatch(r"([0-9a-fA-F]{64})\s+([^\s]+)", checksum)
    if matched is None or matched.group(2) != basename:
        raise ArchiveIntegrityError(f"{basename}: malformed official checksum")
    response = _fetch(fetcher, url)
    digest = sha256(response.body).hexdigest()
    if digest.lower() != matched.group(1).lower():
        raise ArchiveIntegrityError(f"{basename}: SHA-256 mismatch")
    try:
        csv_bytes = _read_zip_csv(response.body, basename)
        if not csv_bytes:
            raise ArchiveIntegrityError(f"{basename}: empty daily CSV")
        header = _csv_header(csv_bytes, kind)
        frame = _parse_complete_bars(
            csv_bytes, kind=kind, symbol=symbol, month=f"{day:%Y-%m}"
        )
    except ValueError as exc:
        raise ArchiveIntegrityError(f"{basename}: {exc}") from exc
    expected = pd.date_range(
        day, day + pd.Timedelta(days=1) - BAR_PERIOD,
        freq=BAR_PERIOD, tz="UTC", name="open_time",
    )
    if not frame.index.equals(expected):
        raise ArchiveIntegrityError(f"{basename}: expected exactly six UTC 4h bars")
    detail = {
        "url": url,
        "checksum_url": url + ".CHECKSUM",
        "sha256": digest,
        "checksum_sha256": sha256(checksum_response.body).hexdigest(),
        "zip_bytes": len(response.body),
        "member": basename[:-4] + ".csv",
        "header": header,
        "rows_read": len(frame),
        "replacement_date": f"{day:%Y-%m-%d}",
        "first": frame.index[0].isoformat(),
        "last": frame.index[-1].isoformat(),
    }
    return frame, detail


def _load_and_audit_impl(
    *, titles: dict[str, str], start: pd.Timestamp, page_end: pd.Timestamp,
    terminal: pd.Timestamp, last_archive_month: str,
    fetcher: Callable[[str], bytes | _Payload], max_workers: int,
) -> DataBundle:
    if max_workers < 1 or max_workers > 12:
        raise ValueError("max_workers must be 1..12")
    if start.tzinfo is None or terminal.tzinfo is None or page_end.tzinfo is None:
        raise ValueError("boundaries must be UTC-aware")
    if start != pd.Timestamp(f"{start:%Y-%m}-01", tz="UTC"):
        raise ValueError("start must be a month start")
    if terminal != pd.Timestamp(f"{terminal:%Y-%m}-01", tz="UTC"):
        raise ValueError("terminal must be a month start")
    if page_end != terminal - pd.Timedelta(days=1):
        raise ValueError("page_end must immediately precede terminal")
    months = _month_range(f"{start:%Y-%m}", last_archive_month)
    if months[-1] != f"{page_end:%Y-%m}":
        raise ValueError("last_archive_month must include page_end")

    views, wiki_manifest = {}, {}
    # Wikimedia requests remain sequential, as in the original loader.
    for symbol, title in titles.items():
        views[symbol], wiki_manifest[symbol] = _load_page(
            title, start, page_end, fetcher
        )

    monthly: dict[tuple[str, str, str], pd.DataFrame] = {}
    archive_manifest: dict[str, dict] = {}
    tasks = [
        (kind, symbol, month, False)
        for symbol in titles for kind in ARCHIVE_KINDS for month in months
    ] + [
        (kind, symbol, f"{terminal:%Y-%m}", True)
        for symbol in titles for kind in ARCHIVE_KINDS
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _load_archive, kind, symbol, month, is_terminal, fetcher, terminal
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

    # Determine every permissible day from missing timestamps alone. A
    # partial-day defect fails without touching a daily source.
    missing: dict[tuple[str, str, str], tuple[pd.Timestamp, ...]] = {}
    for symbol in titles:
        for kind in BAR_KINDS:
            for month in months:
                key = (kind, symbol, month)
                missing[key] = _full_missing_days(
                    monthly[key], kind=kind, symbol=symbol, month=month
                )
                archive_manifest[f"monthly/{kind}/{symbol}/{month}"][
                    "missing_full_utc_days"
                ] = [f"{day:%Y-%m-%d}" for day in missing[key]]
    daily_tasks = [
        (kind, symbol, month, day)
        for symbol in titles for kind in BAR_KINDS for month in months
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
    for (kind, symbol, month), frame in monthly.items():
        segments[(kind, symbol)].append(frame)
    for (kind, symbol, month, day), frame in daily.items():
        segments[(kind, symbol)].append(frame)

    trade_bars, mark_bars, funding_events, coverage = {}, {}, {}, {}
    for symbol in titles:
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
            start=start, terminal=terminal, mark_index=mark_bars[symbol].index,
        )
        coverage[symbol] = {
            "trade": trade_detail,
            "mark": mark_detail,
            "funding": funding_detail,
        }

    replacements = [
        {"kind": kind, "symbol": symbol, "month": month,
         "date": f"{day:%Y-%m-%d}"}
        for kind, symbol, month, day in sorted(daily_tasks)
    ]
    manifest = {
        "wiki_source": WIKIMEDIA_BASE,
        "binance_monthly_source": "https://data.binance.vision/data/futures/um/monthly",
        "binance_daily_source": DAILY_BASE,
        "registered_pages": dict(titles),
        "start": start.isoformat(),
        "page_end": page_end.isoformat(),
        "terminal": terminal.isoformat(),
        "wikipedia": wiki_manifest,
        "archive_count": len(archive_manifest),
        "monthly_archive_count": len(tasks),
        "daily_archive_count": len(daily_tasks),
        "daily_replacements": replacements,
        "archives": dict(sorted(archive_manifest.items())),
        "coverage": coverage,
    }
    json.dumps(manifest, allow_nan=False)
    return DataBundle(views, trade_bars, mark_bars, funding_events, manifest)


def load_and_audit(
    *, fetcher: Callable[[str], bytes | _Payload] | None = None,
    max_workers: int = 8,
) -> DataBundle:
    """Load the fixed five-coin bundle under the separately registered repair.

    A supplied ``fetcher`` enables offline fixture tests; the production
    constants are fixed to the 2021–2025 historical window.
    """
    return _load_and_audit_impl(
        titles=PAGE_TITLES, start=START, page_end=PAGE_END,
        terminal=TERMINAL, last_archive_month=LAST_ARCHIVE_MONTH,
        fetcher=fetcher or _http_fetch, max_workers=max_workers,
    )
