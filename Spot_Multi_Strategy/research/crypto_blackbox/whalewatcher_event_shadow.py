"""Read WhaleWatcher event journal as prospective, non-trading evidence.

This reader never imports WhaleWatcher Python code or writes to its working
tree. It accepts only a fresh local journal, copies no prose, review,
post-event price, prediction probability, notification, or account detail,
and assigns each event the later of publication and Quant's actual read time
as its earliest usable time.
The output cannot establish historical point-in-time availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


MAX_BYTES = 24 * 1024 * 1024
MAX_AGE_MS = 10 * 60_000
MAX_ROWS = 2_000
SOURCES = {
    "bloomberg": {"bloomberg.com", "www.bloomberg.com"},
    "jin10": {"flash.jin10.com", "xnews.jin10.com", "www.jin10.com", "jin10.com"},
    "polymarket": {"polymarket.com", "www.polymarket.com"},
}
CATEGORIES = {"monetary_policy", "inflation_employment", "crypto",
              "regulation", "geopolitics", "us_equities", "security"}


@dataclass(frozen=True)
class ShadowEvent:
    event_sha256: str
    source: str
    category: str
    source_publication_ms: int
    watcher_first_seen_ms: int
    quant_available_ms: int
    source_url_sha256: str
    research_only: bool = field(default=True, init=False)
    execution_eligible: bool = field(default=False, init=False)
    historical_oos_eligible: bool = field(default=False, init=False)


@dataclass(frozen=True)
class EventShadowImport:
    status: str
    captured_at_ms: int
    snapshot_generated_at_ms: int | None = None
    snapshot_sha256: str | None = None
    events: tuple[ShadowEvent, ...] = ()
    research_only: bool = field(default=True, init=False)
    execution_eligible: bool = field(default=False, init=False)
    historical_oos_eligible: bool = field(default=False, init=False)


class _Invalid(ValueError):
    pass


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _Invalid("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise _Invalid("nonfinite JSON constant")


def _timestamp(value, limit: int) -> bool:
    return type(value) is int and 0 < value <= limit


def _url_ok(value, source: str) -> bool:
    if (not isinstance(value, str) or len(value) > 2048
            or any(ord(c) < 32 for c in value) or "\\" in value):
        return False
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == "https" and parsed.hostname in SOURCES[source]
                and parsed.username is None and parsed.password is None
                and parsed.port in (None, 443))
    except ValueError:
        return False


def read_whalewatcher_event_shadow(path: str | Path, *, max_age_ms: int = MAX_AGE_MS
                                    ) -> EventShadowImport:
    """Import fresh event provenance only; all malformed inputs fail closed."""
    if type(max_age_ms) is not int or not 0 < max_age_ms <= MAX_AGE_MS:
        raise ValueError("max_age_ms must be within (0, 600000]")
    captured = time.time_ns() // 1_000_000
    unavailable = EventShadowImport("unavailable", captured)
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        if not raw or len(raw) > MAX_BYTES:
            return unavailable
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                             parse_constant=_reject_nonfinite)
        captured = time.time_ns() // 1_000_000
        unavailable = EventShadowImport("unavailable", captured)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return unavailable
        generated = payload.get("updated_at_ms")
        rows = payload.get("records")
        if (not _timestamp(generated, captured)
                or captured - generated > max_age_ms
                or not isinstance(rows, list) or len(rows) > MAX_ROWS
                or payload.get("execution_eligible") is not False):
            return unavailable
        events = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("event"), dict):
                raise _Invalid("invalid journal row")
            event = row["event"]
            source = event.get("source")
            if source not in SOURCES:
                continue
            identifier = event.get("id")
            category = event.get("category")
            published = event.get("published_at_ms")
            first_seen = row.get("first_seen_at_ms")
            url = event.get("url")
            if (not isinstance(identifier, str) or not 0 < len(identifier) <= 256
                    or identifier in seen or category not in CATEGORIES
                    or not _timestamp(published, captured + 300_000)
                    or not _timestamp(first_seen, generated)
                    or not _url_ok(url, source)):
                raise _Invalid("invalid source event provenance")
            seen.add(identifier)
            events.append(ShadowEvent(
                event_sha256=sha256(identifier.encode("utf-8")).hexdigest(),
                source=source,
                category=category,
                source_publication_ms=published,
                watcher_first_seen_ms=first_seen,
                quant_available_ms=max(captured, published),
                source_url_sha256=sha256(url.encode("utf-8")).hexdigest(),
            ))
        return EventShadowImport("fresh" if events else "fresh_no_events",
                                 captured, generated, sha256(raw).hexdigest(),
                                 tuple(events))
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError, _Invalid):
        return unavailable
