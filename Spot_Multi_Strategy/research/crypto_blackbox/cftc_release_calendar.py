"""Conservative CFTC COT report availability for retrospective source audits.

The usual Friday release is not guaranteed. Known 2023 and 2025 disruptions
are pinned to CFTC's own historical announcements. This is a lower bound on
availability, not proof of an archived first-published data vintage.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable


# https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm
_KNOWN_LATE_RELEASES = {
    date(2023, 1, 31): date(2023, 2, 24),
    date(2023, 2, 7): date(2023, 3, 3),
    date(2023, 2, 14): date(2023, 3, 8),
    date(2023, 2, 21): date(2023, 3, 10),
    date(2023, 2, 28): date(2023, 3, 14),
    date(2023, 3, 7): date(2023, 3, 16),
    date(2023, 3, 14): date(2023, 3, 21),
    date(2025, 9, 30): date(2025, 11, 19),
    date(2025, 10, 7): date(2025, 11, 21),
    date(2025, 10, 14): date(2025, 11, 25),
    date(2025, 10, 21): date(2025, 12, 2),
    date(2025, 10, 28): date(2025, 12, 5),
    date(2025, 11, 4): date(2025, 12, 9),
    date(2025, 11, 10): date(2025, 12, 10),
    date(2025, 11, 18): date(2025, 12, 12),
    date(2025, 11, 25): date(2025, 12, 15),
    date(2025, 12, 2): date(2025, 12, 17),
    date(2025, 12, 9): date(2025, 12, 19),
    date(2025, 12, 16): date(2025, 12, 23),
    date(2025, 12, 23): date(2025, 12, 29),
}


def conservative_eligible_at(report_date: date, *, minimum_lag_days: int = 13) -> datetime:
    """Require a frozen minimum lag and, for known delays, a full publication day.

    The default 13 days is still a proxy for unlisted exceptional delays.
    These archives lack first-published vintages and cannot certify true OOS.
    """
    if type(report_date) is not date:
        raise ValueError("report_date must be a calendar date")
    if type(minimum_lag_days) is not int or not 1 <= minimum_lag_days <= 30:
        raise ValueError("minimum_lag_days must be a fixed integer from 1 to 30")
    normal = datetime.combine(report_date + timedelta(days=minimum_lag_days),
                              time.min, timezone.utc)
    published = _KNOWN_LATE_RELEASES.get(report_date)
    if published is None:
        return normal
    buffer = datetime.combine(published + timedelta(days=1), time.min, timezone.utc)
    return max(normal, buffer)


def latest_eligible_report(report_dates: Iterable[date], decision_at: datetime,
                           *, minimum_lag_days: int = 13) -> date | None:
    """Select the latest report whose availability bound precedes a UTC decision."""
    if decision_at.tzinfo is None or decision_at.utcoffset() != timedelta(0):
        raise ValueError("decision_at must be UTC aware")
    dates = list(report_dates)
    if dates != sorted(set(dates)):
        raise ValueError("report dates must be strictly increasing and unique")
    available = [day for day in dates
                 if conservative_eligible_at(day, minimum_lag_days=minimum_lag_days)
                 <= decision_at]
    return available[-1] if available else None
