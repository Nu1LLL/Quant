"""News-only prospective features; claim-feed outages cannot erase valid news.

These counts contain no prices, direction, article prose, recommendations or
order instructions. They are observation inputs for a future preregistration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from types import MappingProxyType
from typing import Mapping

from .whalewatcher_event_shadow import CATEGORIES, SOURCES
from .whalewatcher_prospective_journal import (
    MAX_POLL_GAP_MS, poll_coverage, read_events_asof,
)


HOUR_MS = 3_600_000
MAX_PUBLICATION_TO_QUANT_MS = 2 * HOUR_MS
NEWS_FEATURE_NAMES = tuple(
    [f"event_{category}_{label}" for label in ("1h", "24h")
     for category in sorted(CATEGORIES)]
    + [f"event_source_{source}_24h" for source in sorted(SOURCES)]
)


@dataclass(frozen=True)
class NewsFeatureVector:
    asof_ms: int
    source_available: bool
    values: Mapping[str, int]
    research_only: bool = field(default=True, init=False)
    execution_eligible: bool = field(default=False, init=False)
    historical_oos_eligible: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.asof_ms) is not int or self.asof_ms <= 0:
            raise ValueError("asof_ms must be a positive integer")
        if type(self.source_available) is not bool:
            raise TypeError("source_available must be boolean")
        if self.source_available != (set(self.values) == set(NEWS_FEATURE_NAMES)):
            raise ValueError("available news vectors need the fixed schema")
        if any(type(value) is not int or value < 0 for value in self.values.values()):
            raise ValueError("news feature counts must be nonnegative integers")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


def build_news_features(journal_path: str | Path, asof_ms: int) -> NewsFeatureVector:
    if type(asof_ms) is not int or asof_ms <= 0:
        raise ValueError("asof_ms must be a positive integer")
    if asof_ms > time.time_ns() // 1_000_000:
        raise ValueError("cannot build news features for a future decision")
    coverage = poll_coverage(journal_path, max(1, asof_ms - MAX_POLL_GAP_MS),
                             asof_ms, stream="events")
    available = (coverage.polls > 0 and coverage.unavailable_polls == 0
                 and coverage.largest_gap_ms <= MAX_POLL_GAP_MS)
    if not available:
        return NewsFeatureVector(asof_ms, False, {})
    values = {name: 0 for name in NEWS_FEATURE_NAMES}
    for event in read_events_asof(journal_path, asof_ms):
        if event.category not in CATEGORIES or event.source not in SOURCES:
            raise ValueError("unknown news source or category in journal")
        if event.quant_available_ms > asof_ms:
            raise ValueError("future news leaked into as-of snapshot")
        if not 0 <= event.quant_available_ms - event.source_publication_ms <= MAX_PUBLICATION_TO_QUANT_MS:
            continue
        age = asof_ms - event.source_publication_ms
        if age < 0:
            raise ValueError("future news publication leaked")
        if age <= HOUR_MS:
            values[f"event_{event.category}_1h"] += 1
        if age <= 24 * HOUR_MS:
            values[f"event_{event.category}_24h"] += 1
            values[f"event_source_{event.source}_24h"] += 1
    return NewsFeatureVector(asof_ms, True, values)
