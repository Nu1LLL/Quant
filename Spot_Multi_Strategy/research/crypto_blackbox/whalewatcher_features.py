"""Point-in-time, non-directional WhaleWatcher features for future research.

Only Quant's actual observation time starts an event window. A missing source
poll is missing data, never a zero-news observation. No returns, text, advice,
orders or private URLs enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from types import MappingProxyType
from typing import Mapping

from .whalewatcher_event_shadow import CATEGORIES, SOURCES
from .whalewatcher_prospective_journal import (
    MAX_POLL_GAP_MS, poll_coverage, read_asof,
)


HOUR_MS = 3_600_000
MAX_PUBLICATION_TO_QUANT_MS = 2 * HOUR_MS
WINDOWS = (("1h", HOUR_MS), ("24h", 24 * HOUR_MS))
FEATURE_NAMES = tuple(
    [f"event_{category}_{label}" for label, _ in WINDOWS
     for category in sorted(CATEGORIES)]
    + [f"event_source_{source}_24h" for source in sorted(SOURCES)]
    + [f"claim_{symbol}_{side}_24h" for symbol in ("BTC", "ETH")
       for side in ("long", "short")]
)


@dataclass(frozen=True)
class EventFeatureVector:
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
        if self.source_available != (set(self.values) == set(FEATURE_NAMES)):
            raise ValueError("available vectors need exactly the fixed feature schema")
        if any(type(value) is not int or value < 0 for value in self.values.values()):
            raise ValueError("feature counts must be nonnegative integers")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


def build_event_features(journal_path: str | Path, asof_ms: int) -> EventFeatureVector:
    """Build an as-of feature vector only when a recent source poll exists."""
    if type(asof_ms) is not int or asof_ms <= 0:
        raise ValueError("asof_ms must be a positive integer")
    if asof_ms > time.time_ns() // 1_000_000:
        raise ValueError("cannot build features for a future decision")
    coverage = poll_coverage(journal_path, max(1, asof_ms - MAX_POLL_GAP_MS), asof_ms)
    source_available = (coverage.polls > 0 and coverage.unavailable_polls == 0
                        and coverage.largest_gap_ms <= MAX_POLL_GAP_MS)
    if not source_available:
        return EventFeatureVector(asof_ms, False, {})
    snapshot = read_asof(journal_path, asof_ms)
    values = {name: 0 for name in FEATURE_NAMES}
    for event in snapshot.events:
        if event.category not in CATEGORIES or event.source not in SOURCES:
            raise ValueError("unknown event source or category in journal")
        if event.quant_available_ms > asof_ms:
            raise ValueError("future event leaked into as-of snapshot")
        if not 0 <= event.quant_available_ms - event.source_publication_ms <= MAX_PUBLICATION_TO_QUANT_MS:
            continue  # A delayed first import is not a fresh news arrival.
        age = asof_ms - event.source_publication_ms
        for label, width in WINDOWS:
            if age <= width:
                values[f"event_{event.category}_{label}"] += 1
        if age <= 24 * HOUR_MS:
            values[f"event_source_{event.source}_24h"] += 1
    for symbol, claims in (("BTC", snapshot.btc_claims), ("ETH", snapshot.eth_claims)):
        for claim in claims:
            if claim.asserted_side not in {"long", "short"}:
                raise ValueError("unknown claim side in journal")
            if claim.quant_available_at_ms > asof_ms:
                raise ValueError("future claim leaked into as-of snapshot")
            if (0 <= claim.quant_available_at_ms - claim.source_available_at_ms
                    <= MAX_PUBLICATION_TO_QUANT_MS
                    and asof_ms - claim.source_available_at_ms <= 24 * HOUR_MS):
                values[f"claim_{symbol}_{claim.asserted_side}_24h"] += 1
    return EventFeatureVector(asof_ms, True, values)
