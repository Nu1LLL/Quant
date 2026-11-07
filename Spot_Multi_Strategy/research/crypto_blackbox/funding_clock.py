"""Map Binance funding settlement timestamps to completed UTC clock hours.

Official archived ``calc_time`` values often carry a few milliseconds after
the hour. This strict mapping is source preprocessing, never a trading signal.
"""

from __future__ import annotations

from datetime import datetime, timezone


HOUR_MS = 3_600_000
MAX_POST_HOUR_MS = 999


class FundingClockError(ValueError):
    pass


def settled_hour_ms(raw_ms: int) -> int:
    """Return the UTC settlement hour for a source stamp within 999 ms of it."""
    if type(raw_ms) is not int or raw_ms <= 0:
        raise FundingClockError("funding timestamp must be a positive integer millisecond")
    offset = raw_ms % HOUR_MS
    if offset > MAX_POST_HOUR_MS:
        raise FundingClockError("funding timestamp is not within 999 ms after a UTC hour")
    return raw_ms - offset


def settled_hour(raw_ms: int) -> datetime:
    return datetime.fromtimestamp(settled_hour_ms(raw_ms) / 1000, timezone.utc)


def unique_settlements(raw_stamps: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Reject duplicate or out-of-order settlement hours rather than netting."""
    anchors = tuple(settled_hour_ms(stamp) for stamp in raw_stamps)
    if any(later <= earlier for earlier, later in zip(anchors, anchors[1:])):
        raise FundingClockError("funding settlements are duplicated or out of order")
    return anchors
