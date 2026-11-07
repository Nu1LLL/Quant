"""Prospective, read-only bridge for WhaleWatcher's official macro schedule.

Only event identity hashes, schedule timestamps and source provenance enter
this small append-only journal. No AI forecasts, economic releases, prices,
market odds, advice, orders or performance enter it. A historical query can
see a schedule only after this collector actually read it.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urlsplit


MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_SOURCE_AGE_MS = 10 * 60_000
MAX_ROWS = 100
MAX_POLL_GAP_MS = 2 * 60 * 60_000
MIN_OOS_MS = 5 * 365 * 24 * 60 * 60_000
KINDS = {"cpi": "www.bls.gov", "nfp": "www.bls.gov",
         "pce": "www.bea.gov", "fomc": "www.federalreserve.gov"}
PROVENANCE = re.compile(r"official_live|official_snapshot_\d{4}_\d{2}_\d{2}\Z")


@dataclass(frozen=True)
class MacroEvent:
    identity_sha256: str
    kind: str
    scheduled_at_ms: int
    schedule_checked_at_ms: int
    schedule_time_confirmed: bool
    schedule_provenance: str
    url_sha256: str
    quant_available_at_ms: int
    withdrawn: bool
    execution_eligible: bool = False
    historical_oos_eligible: bool = False


@dataclass(frozen=True)
class MacroCapture:
    status: str
    observed_at_ms: int
    active_events: int
    changed_events: int
    execution_eligible: bool = False
    historical_oos_eligible: bool = False


@dataclass(frozen=True)
class MacroCoverage:
    start_ms: int
    end_ms: int
    polls: int
    unavailable_polls: int
    largest_gap_ms: int
    five_year_continuous_source: bool
    execution_eligible: bool = False


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate WhaleWatcher JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _valid_time(value, maximum: int) -> bool:
    return type(value) is int and 0 < value <= maximum


def _schedule_url(value, kind: str) -> bool:
    if not isinstance(value, str) or len(value) > 2048 or any(ord(c) < 32 for c in value):
        return False
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == "https" and parsed.hostname == KINDS[kind]
                and parsed.username is None and parsed.password is None
                and parsed.port in (None, 443))
    except ValueError:
        return False


def _read(path: Path, observed: int) -> tuple[str, int | None, str | None,
                                                dict[str, tuple]]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_SOURCE_BYTES + 1)
        if not raw or len(raw) > MAX_SOURCE_BYTES:
            raise ValueError("missing or oversized WhaleWatcher macro snapshot")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                             parse_constant=_reject_constant)
        if not isinstance(payload, dict):
            raise ValueError("macro snapshot is not an object")
        generated = payload.get("updated_at_ms")
        rows = payload.get("records")
        if (payload.get("schema_version") != 1 or
                not _valid_time(generated, observed) or
                observed - generated > MAX_SOURCE_AGE_MS or
                not isinstance(rows, list) or len(rows) > MAX_ROWS):
            raise ValueError("stale or invalid WhaleWatcher macro snapshot")
        incoming = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("event"), dict):
                raise ValueError("invalid macro record")
            event = row["event"]
            kind = event.get("kind")
            if kind not in KINDS or event.get("source") != "macro_calendar":
                raise ValueError("unrecognized macro source")
            identifier = event.get("id")
            scheduled = event.get("scheduled_at_ms")
            checked = event.get("schedule_checked_at_ms")
            confirmed = event.get("schedule_time_confirmed")
            provenance = event.get("schedule_provenance")
            url = event.get("schedule_url")
            withdrawn = row.get("schedule_withdrawn")
            if (not isinstance(identifier, str) or not 0 < len(identifier) <= 256
                    or not _valid_time(scheduled, observed + 2 * 366 * 86_400_000)
                    or scheduled < observed - 366 * 86_400_000
                    or not _valid_time(checked, observed)
                    or type(confirmed) is not bool or type(withdrawn) is not bool
                    or not isinstance(provenance, str)
                    or PROVENANCE.fullmatch(provenance) is None
                    or event.get("time_basis") != "scheduled"
                    or not _schedule_url(url, kind)):
                raise ValueError("invalid macro schedule provenance")
            identity = sha256(identifier.encode("utf-8")).hexdigest()
            if identity in incoming:
                raise ValueError("duplicate macro event identity")
            incoming[identity] = (kind, scheduled, checked, int(confirmed),
                                  provenance, sha256(url.encode("utf-8")).hexdigest(),
                                  int(withdrawn))
        return "fresh", generated, sha256(raw).hexdigest(), incoming
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        return "unavailable", None, None, {}


def _initialize(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS polls (
            id INTEGER PRIMARY KEY,
            observed_at_ms INTEGER NOT NULL CHECK(observed_at_ms > 0),
            status TEXT NOT NULL CHECK(status IN ('fresh','unavailable')),
            generated_at_ms INTEGER,
            snapshot_sha256 TEXT,
            CHECK((status='fresh' AND generated_at_ms IS NOT NULL
                   AND snapshot_sha256 IS NOT NULL AND length(snapshot_sha256)=64)
               OR (status='unavailable' AND generated_at_ms IS NULL
                   AND snapshot_sha256 IS NULL))
        );
        CREATE INDEX IF NOT EXISTS polls_asof ON polls(observed_at_ms,id);
        CREATE TABLE IF NOT EXISTS changes (
            id INTEGER PRIMARY KEY,
            poll_id INTEGER NOT NULL REFERENCES polls(id),
            identity_sha256 TEXT NOT NULL CHECK(length(identity_sha256)=64),
            operation TEXT NOT NULL CHECK(operation IN ('upsert','remove')),
            kind TEXT,
            scheduled_at_ms INTEGER,
            schedule_checked_at_ms INTEGER,
            confirmed INTEGER,
            provenance TEXT,
            url_sha256 TEXT,
            withdrawn INTEGER,
            quant_available_at_ms INTEGER NOT NULL,
            UNIQUE(poll_id,identity_sha256)
        );
        CREATE INDEX IF NOT EXISTS changes_asof ON changes(quant_available_at_ms,id);
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
    params = () if asof_ms is None else (asof_ms,)
    condition = "WHERE c.quant_available_at_ms<=?" if asof_ms is not None else ""
    result = {}
    for row in db.execute(
        "SELECT c.identity_sha256,c.operation,c.kind,c.scheduled_at_ms,"
        "c.schedule_checked_at_ms,c.confirmed,c.provenance,c.url_sha256,"
        "c.withdrawn,c.quant_available_at_ms FROM changes c "
        f"{condition} ORDER BY c.id", params,
    ):
        identity, operation, *metadata = row
        if operation == "remove":
            result.pop(identity, None)
        else:
            result[identity] = tuple(metadata)
    return result


def capture_current(watcher_data_dir: str | Path, journal_path: str | Path,
                    *, observed_at_ms: int | None = None) -> MacroCapture:
    """Append the current macro schedule's *metadata changes* and a poll."""
    observed = time.time_ns() // 1_000_000 if observed_at_ms is None else observed_at_ms
    if type(observed) is not int or observed <= 0:
        raise ValueError("invalid observation clock")
    source = Path(watcher_data_dir) / "event_state" / "macro.json"
    path = Path(journal_path)
    if source.resolve() == path.resolve():
        raise ValueError("journal cannot replace a WhaleWatcher snapshot")
    status, generated, digest, incoming = _read(source, observed)
    if status == "unavailable" and not path.is_file():
        return MacroCapture(status, observed, 0, 0)
    if path.is_file() and path.stat().st_size > MAX_JOURNAL_BYTES:
        raise RuntimeError("macro journal reached its 16 MiB storage cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        _initialize(db)
        with db:
            previous = db.execute("SELECT observed_at_ms FROM polls ORDER BY id DESC LIMIT 1").fetchone()
            if previous is not None and observed <= previous[0]:
                raise ValueError("macro poll clock did not advance")
            cursor = db.execute(
                "INSERT INTO polls(observed_at_ms,status,generated_at_ms,snapshot_sha256) "
                "VALUES (?,?,?,?)", (observed, status, generated, digest))
            changed = 0
            if status == "fresh":
                prior = _state(db)
                for identity in sorted(prior.keys() - incoming.keys()):
                    db.execute("INSERT INTO changes(poll_id,identity_sha256,operation,"
                               "quant_available_at_ms) VALUES (?,?,?,?)",
                               (cursor.lastrowid, identity, "remove", observed))
                    changed += 1
                for identity, metadata in sorted(incoming.items()):
                    if prior.get(identity, ())[:-1] == metadata:
                        continue
                    db.execute("INSERT INTO changes VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)",
                               (cursor.lastrowid, identity, "upsert", *metadata, observed))
                    changed += 1
            pages = db.execute("PRAGMA page_count").fetchone()[0]
            page_size = db.execute("PRAGMA page_size").fetchone()[0]
            if pages * page_size > MAX_JOURNAL_BYTES:
                raise RuntimeError("macro journal reached its 16 MiB storage cap")
    return MacroCapture(status, observed, len(incoming) if status == "fresh" else 0, changed)


def read_asof(journal_path: str | Path, asof_ms: int) -> tuple[MacroEvent, ...]:
    """Recover only schedule metadata already captured by this Quant journal."""
    if type(asof_ms) is not int or asof_ms <= 0:
        raise ValueError("invalid as-of time")
    path = Path(journal_path)
    if not path.is_file():
        return ()
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        state = _state(db, asof_ms)
    return tuple(MacroEvent(identity, metadata[0], metadata[1], metadata[2],
                            bool(metadata[3]), metadata[4], metadata[5],
                            metadata[7], bool(metadata[6]))
                 for identity, metadata in sorted(state.items()))


def poll_coverage(journal_path: str | Path, start_ms: int, end_ms: int) -> MacroCoverage:
    """Measure source continuity; this alone never proves trading Alpha."""
    if (type(start_ms) is not int or type(end_ms) is not int
            or not 0 < start_ms < end_ms):
        raise ValueError("coverage requires increasing positive UTC milliseconds")
    path = Path(journal_path)
    if not path.is_file():
        return MacroCoverage(start_ms, end_ms, 0, 0, end_ms - start_ms, False)
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        rows = db.execute(
            "SELECT observed_at_ms,status FROM polls "
            "WHERE observed_at_ms>=? AND observed_at_ms<=? "
            "ORDER BY observed_at_ms,id", (start_ms, end_ms)).fetchall()
    stamps = [start_ms, *(int(row[0]) for row in rows), end_ms]
    largest = max(b - a for a, b in zip(stamps, stamps[1:]))
    unavailable = sum(row[1] != "fresh" for row in rows)
    continuous = (end_ms - start_ms >= MIN_OOS_MS and bool(rows)
                  and largest <= MAX_POLL_GAP_MS and unavailable == 0)
    return MacroCoverage(start_ms, end_ms, len(rows), unavailable, largest, continuous)


def upcoming_confirmed(journal_path: str | Path, asof_ms: int,
                       *, within_days: int = 7) -> tuple[MacroEvent, ...]:
    """Describe future confirmed schedules; never emit trade directions."""
    if type(within_days) is not int or not 0 < within_days <= 30:
        raise ValueError("within_days must be 1..30")
    return tuple(sorted((event for event in read_asof(journal_path, asof_ms)
                         if event.schedule_time_confirmed and not event.withdrawn
                         and asof_ms < event.scheduled_at_ms <= asof_ms + within_days * 86_400_000),
                        key=lambda event: (event.scheduled_at_ms, event.kind)))
