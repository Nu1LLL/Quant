"""Checksum-verified Binance BTCUSDT daily spot bars for the CME COT study.

The preregistered source is the official monthly 1d spot archive from
2018-12 through 2026-01.  ZIP and CSV bytes stay in bounded memory.  Every
month is validated before a bar is returned; only the 2026-01-01 opening bar
is retained from the final month for the forced liquidation valuation.

Parsing, official CHECKSUM verification, and bounded HTTP downloads reuse the
existing BTC difficulty archive implementation.  This module fixes a separate
study window and requires a complete daily grid within *each* archive.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

if __package__:
    from . import btc_difficulty_spot as _archive
else:
    import btc_difficulty_spot as _archive


UTC = timezone.utc
START = datetime(2018, 12, 1, tzinfo=UTC)
LAST_OPEN = datetime(2026, 1, 1, tzinfo=UTC)
END_EXCLUSIVE = LAST_OPEN + timedelta(days=1)
FIRST_MONTH = "2018-12"
LAST_MONTH = "2026-01"

ArchiveIntegrityError = _archive.ArchiveIntegrityError
ArchiveUnavailable = _archive.ArchiveUnavailable
Bar = _archive.Bar
MonthManifest = _archive.MonthManifest
MonthArchive = _archive.MonthArchive
BarsResult = _archive.BarsResult
archive_url = _archive.archive_url


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    """Return UTC boundaries for a month in this frozen archive window."""
    archive_url(month)  # Validate the exact YYYY-MM archive path component.
    if not FIRST_MONTH <= month <= LAST_MONTH:
        raise ValueError(f"month outside CME BTC study window: {month}")
    year, number = map(int, month.split("-"))
    start = datetime(year, number, 1, tzinfo=UTC)
    end = (datetime(year + 1, 1, 1, tzinfo=UTC) if number == 12 else
           datetime(year, number + 1, 1, tzinfo=UTC))
    return start, end


def _months() -> tuple[str, ...]:
    """All required official monthly archives, including endpoint January."""
    months: list[str] = []
    year, number = START.year, START.month
    while (year, number) <= (LAST_OPEN.year, LAST_OPEN.month):
        months.append(f"{year:04d}-{number:02d}")
        year, number = (year + 1, 1) if number == 12 else (year, number + 1)
    return tuple(months)


def fetch_and_parse_month(
    month: str, *, fetcher: Callable[[str], bytes] | None = None
) -> MonthArchive:
    """Verify one official ZIP/CHECKSUM pair and its complete UTC daily grid.

    ``fetcher`` is an injectable byte fetcher for offline synthetic fixtures.
    No raw archive or CSV is written to disk.  An unavailable or malformed
    archive raises ``ArchiveUnavailable`` or ``ArchiveIntegrityError``.
    """
    start, end = _month_bounds(month)
    parsed = _archive.fetch_and_parse_month(month, fetcher=fetcher)
    expected_open = start
    for bar in parsed.bars:
        if bar.open_time != expected_open:
            raise ArchiveIntegrityError(
                f"BTCUSDT {month} 1d grid gap/duplicate: expected "
                f"{expected_open.isoformat()}, received {bar.open_time.isoformat()}"
            )
        expected_open += timedelta(days=1)
    if expected_open != end:
        raise ArchiveIntegrityError(
            f"BTCUSDT {month} 1d grid incomplete: expected bar at "
            f"{expected_open.isoformat()}"
        )
    return parsed


def fetch_bars(*, fetcher: Callable[[str], bytes] | None = None) -> BarsResult:
    """Load the frozen daily bar grid through the 2026-01-01 open.

    The final month is fully audited, but bars after January 1 are excluded
    because the study's position is forcibly closed at that day's open.
    """
    bars: list[Bar] = []
    manifest: list[MonthManifest] = []
    expected_open = START
    for month in _months():
        parsed = fetch_and_parse_month(month, fetcher=fetcher)
        manifest.append(parsed.manifest)
        for bar in parsed.bars:
            if bar.open_time >= END_EXCLUSIVE:
                continue
            if bar.open_time != expected_open:
                raise ArchiveIntegrityError(
                    f"BTCUSDT 1d grid gap/duplicate: expected "
                    f"{expected_open.isoformat()}, received {bar.open_time.isoformat()}"
                )
            bars.append(bar)
            expected_open += timedelta(days=1)
    if expected_open != END_EXCLUSIVE:
        raise ArchiveIntegrityError(
            f"BTCUSDT 1d grid incomplete: expected bar at "
            f"{expected_open.isoformat()}"
        )
    return BarsResult(tuple(bars), tuple(manifest))


__all__ = [
    "ArchiveIntegrityError", "ArchiveUnavailable", "Bar", "MonthManifest",
    "MonthArchive", "BarsResult", "START", "LAST_OPEN", "END_EXCLUSIVE",
    "FIRST_MONTH", "LAST_MONTH", "archive_url", "fetch_and_parse_month",
    "fetch_bars",
]
