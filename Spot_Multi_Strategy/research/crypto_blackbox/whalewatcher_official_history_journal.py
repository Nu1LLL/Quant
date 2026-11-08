"""Value-blind, append-only bridge for WhaleWatcher official macro history.

Historical publication dates are useful retrospective event metadata. They
never become independent OOS merely because WhaleWatcher later backfilled or
verified an archive. Quant's first read/revision time is retained separately.
No article prose, macro value, price, AI opinion, or order is copied.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit


MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_EVENTS = 2_000
KINDS = {"cpi": ("bls", "www.bls.gov"),
         "nfp": ("bls", "www.bls.gov"),
         "pce": ("bea", "www.bea.gov")}
TIME_BASIS = frozenset({"official_archive_embargo_header",
                        "official_archive_embargo_header_snapshot"})
TIMEZONES = {"EDT": -240, "EST": -300}


@dataclass(frozen=True)
class HistoryEvent:
    identity_sha256: str
    kind: str
    released_at_ms: int
    source_verified_at_ms: int
    url_sha256: str
    quant_first_seen_at_ms: int
    quant_version_seen_at_ms: int
    retrospective_archive: bool = field(default=True, init=False)
    historical_oos_eligible: bool = field(default=False, init=False)
    execution_eligible: bool = field(default=False, init=False)


@dataclass(frozen=True)
class Capture:
    status: str
    observed_at_ms: int
    active_events: int
    changes: int
    source_sha256: str | None
    historical_oos_eligible: bool = field(default=False, init=False)
    execution_eligible: bool = field(default=False, init=False)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate source JSON key")
        value[key] = item
    return value


def _reject_constant(_value):
    raise ValueError("nonfinite source JSON number")


def _positive_ms(value, upper: int) -> bool:
    return type(value) is int and 0 < value <= upper


def _official_url(value: str, kind: str) -> bool:
    if (not isinstance(value, str) or not 0 < len(value) <= 2048
            or any(ord(char) < 32 for char in value) or "\\" in value):
        return False
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == "https" and parsed.hostname == KINDS[kind][1]
                and parsed.username is None and parsed.password is None
                and parsed.port in (None, 443))
    except ValueError:
        return False


def _source(path: Path, observed: int) -> tuple[str, str | None, dict[str, tuple]]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_SOURCE_BYTES + 1)
        if not raw or len(raw) > MAX_SOURCE_BYTES:
            raise ValueError("missing or oversized official macro archive")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                             parse_constant=_reject_constant)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("unexpected archive schema")
        rows = payload.get("events")
        if not isinstance(rows, list) or len(rows) > MAX_EVENTS:
            raise ValueError("unexpected archive event count")
        incoming = {}
        for event in rows:
            if not isinstance(event, dict):
                raise ValueError("invalid archive event")
            kind = event.get("kind")
            if kind not in KINDS:
                raise ValueError("unrecognized official macro kind")
            identifier = event.get("id")
            released = event.get("occurred_at_ms")
            verified = event.get("verified_at_ms")
            timezone = event.get("release_timezone")
            if (not isinstance(identifier, str) or not 0 < len(identifier) <= 256
                    or not _positive_ms(released, observed)
                    or not _positive_ms(verified, observed)
                    or verified < released
                    or event.get("published_at_ms") != released
                    or event.get("source") != "verified_calendar"
                    or event.get("publisher") != KINDS[kind][0]
                    or event.get("category") != "inflation_employment"
                    or event.get("importance") != "high"
                    or event.get("timestamp_verified") is not True
                    or event.get("time_basis") != "occurred"
                    or event.get("timestamp_basis") not in TIME_BASIS
                    or timezone not in TIMEZONES
                    or event.get("release_utc_offset_minutes") != TIMEZONES[timezone]
                    or not isinstance(event.get("release_time_evidence"), str)
                    or not event["release_time_evidence"].strip()
                    or not _official_url(event.get("url"), kind)):
                raise ValueError("unverified or malformed macro event provenance")
            identity = sha256(identifier.encode("utf-8")).hexdigest()
            if identity in incoming:
                raise ValueError("duplicate official macro event")
            incoming[identity] = (kind, released, verified,
                                  sha256(event["url"].encode("utf-8")).hexdigest())
        return "fresh", sha256(raw).hexdigest(), incoming
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        return "unavailable", None, {}


def _initialize(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS polls (
            id INTEGER PRIMARY KEY,
            observed_at_ms INTEGER NOT NULL CHECK(observed_at_ms>0),
            status TEXT NOT NULL CHECK(status IN ('fresh','unavailable')),
            source_sha256 TEXT,
            event_count INTEGER NOT NULL CHECK(event_count>=0)
        );
        CREATE INDEX IF NOT EXISTS polls_asof ON polls(observed_at_ms,id);
        CREATE TABLE IF NOT EXISTS changes (
            id INTEGER PRIMARY KEY,
            poll_id INTEGER NOT NULL REFERENCES polls(id),
            identity_sha256 TEXT NOT NULL CHECK(length(identity_sha256)=64),
            operation TEXT NOT NULL CHECK(operation IN ('upsert','remove')),
            kind TEXT, released_at_ms INTEGER, source_verified_at_ms INTEGER,
            url_sha256 TEXT, quant_first_seen_at_ms INTEGER NOT NULL,
            quant_version_seen_at_ms INTEGER NOT NULL,
            UNIQUE(poll_id,identity_sha256)
        );
        CREATE INDEX IF NOT EXISTS changes_asof ON changes(quant_version_seen_at_ms,id);
        CREATE TRIGGER IF NOT EXISTS polls_no_update BEFORE UPDATE ON polls
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS polls_no_delete BEFORE DELETE ON polls
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS changes_no_update BEFORE UPDATE ON changes
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS changes_no_delete BEFORE DELETE ON changes
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
    """)


def _state(db: sqlite3.Connection, asof_ms: int | None = None) -> dict[str, tuple]:
    where = "WHERE quant_version_seen_at_ms<=?" if asof_ms is not None else ""
    params = (asof_ms,) if asof_ms is not None else ()
    state = {}
    for row in db.execute(
        "SELECT identity_sha256,operation,kind,released_at_ms,"
        "source_verified_at_ms,url_sha256,quant_first_seen_at_ms,"
        f"quant_version_seen_at_ms FROM changes {where} ORDER BY id", params
    ):
        identity, operation, *metadata = row
        if operation == "remove":
            state.pop(identity, None)
        else:
            state[identity] = tuple(metadata)
    return state


def capture_current(source_path: str | Path, journal_path: str | Path,
                    *, observed_at_ms: int | None = None) -> Capture:
    """Record when Quant first saw the archive and every later revision."""
    observed = time.time_ns() // 1_000_000 if observed_at_ms is None else observed_at_ms
    if type(observed) is not int or observed <= 0:
        raise ValueError("invalid Quant observation clock")
    source, path = Path(source_path), Path(journal_path)
    if source.resolve() == path.resolve():
        raise ValueError("journal cannot replace the WhaleWatcher source")
    status, digest, incoming = _source(source, observed)
    if status == "unavailable" and not path.is_file():
        return Capture(status, observed, 0, 0, None)
    if path.is_file() and path.stat().st_size > MAX_JOURNAL_BYTES:
        raise RuntimeError("official macro journal reached 16 MiB cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        _initialize(db)
        with db:
            prior_poll = db.execute(
                "SELECT observed_at_ms FROM polls ORDER BY id DESC LIMIT 1").fetchone()
            if prior_poll is not None and observed <= prior_poll[0]:
                raise ValueError("Quant poll clock did not advance")
            poll = db.execute(
                "INSERT INTO polls(observed_at_ms,status,source_sha256,event_count) "
                "VALUES (?,?,?,?)", (observed, status, digest, len(incoming)))
            changes = 0
            if status == "fresh":
                previous = _state(db)
                for identity in sorted(previous.keys() - incoming.keys()):
                    old = previous[identity]
                    db.execute("INSERT INTO changes VALUES "
                               "(NULL,?,?,'remove',NULL,NULL,NULL,NULL,?,?)",
                               (poll.lastrowid, identity, old[4], observed))
                    changes += 1
                for identity, metadata in sorted(incoming.items()):
                    old = previous.get(identity)
                    if old is not None and old[:4] == metadata:
                        continue
                    if old is not None:
                        first_seen = old[4]
                    else:
                        prior_first = db.execute(
                            "SELECT MIN(quant_first_seen_at_ms) FROM changes "
                            "WHERE identity_sha256=? AND operation='upsert'",
                            (identity,),
                        ).fetchone()[0]
                        first_seen = prior_first if prior_first is not None else observed
                    db.execute("INSERT INTO changes VALUES "
                               "(NULL,?,?,'upsert',?,?,?,?,?,?)",
                               (poll.lastrowid, identity, *metadata, first_seen, observed))
                    changes += 1
            pages = db.execute("PRAGMA page_count").fetchone()[0]
            page_size = db.execute("PRAGMA page_size").fetchone()[0]
            if pages * page_size > MAX_JOURNAL_BYTES:
                raise RuntimeError("official macro journal reached 16 MiB cap")
    return Capture(status, observed, len(incoming), changes, digest)


def read_asof(journal_path: str | Path, asof_ms: int) -> tuple[HistoryEvent, ...]:
    """Only versions already captured by Quant are visible at this as-of."""
    if type(asof_ms) is not int or asof_ms <= 0:
        raise ValueError("invalid as-of time")
    path = Path(journal_path)
    if not path.is_file():
        return ()
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        state = _state(db, asof_ms)
    return tuple(HistoryEvent(identity, *metadata)
                 for identity, metadata in sorted(state.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Record value-blind official macro history")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    result = capture_current(args.source, args.journal)
    print(json.dumps({"status": result.status,
                      "observed_at_ms": result.observed_at_ms,
                      "active_events": result.active_events,
                      "changes": result.changes,
                      "historical_oos_eligible": False,
                      "execution_eligible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
