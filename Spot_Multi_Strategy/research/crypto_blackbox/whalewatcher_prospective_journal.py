"""Append only WhaleWatcher provenance *changes* for prospective shadow research.

No WhaleWatcher code, raw article text/URL, prices, performance or trading
instruction is imported. Historical queries see a change only after Quant's
actual snapshot-read time. Capturing the same event set again stores nothing.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from .whalewatcher_event_shadow import EventShadowImport, read_whalewatcher_event_shadow
from .whalewatcher_macro_journal import capture_current as capture_macro_current
from .whalewatcher_shadow_adapter import ShadowImport, read_whalewatcher_shadow_snapshot


MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_POLL_GAP_MS = 2 * 60 * 60 * 1_000
MIN_OOS_MS = 5 * 365 * 24 * 60 * 60 * 1_000


@dataclass(frozen=True)
class CaptureStatus:
    kind: str
    symbol: str
    status: str
    count: int
    captured_at_ms: int
    execution_eligible: bool = False
    historical_oos_eligible: bool = False


@dataclass(frozen=True)
class JournalEvent:
    event_sha256: str
    source: str
    category: str
    source_publication_ms: int
    watcher_first_seen_ms: int
    quant_available_ms: int
    source_url_sha256: str
    execution_eligible: bool = False
    historical_oos_eligible: bool = False


@dataclass(frozen=True)
class JournalClaim:
    observation_id: str
    source_group_id: str
    fact_group_sha256: str
    asserted_side: str
    source_available_at_ms: int
    quant_available_at_ms: int
    execution_eligible: bool = False
    historical_oos_eligible: bool = False


@dataclass(frozen=True)
class ProvenanceSnapshot:
    asof_ms: int
    events: tuple[JournalEvent, ...]
    btc_claims: tuple[JournalClaim, ...]
    eth_claims: tuple[JournalClaim, ...]
    research_only: bool = True
    execution_eligible: bool = False
    historical_oos_eligible: bool = False


@dataclass(frozen=True)
class PollCoverage:
    """Source-observation continuity, never an Alpha or trading gate pass."""

    start_ms: int
    end_ms: int
    polls: int
    unavailable_polls: int
    largest_gap_ms: int
    five_year_continuous_source: bool
    execution_eligible: bool = False


def _initialize(db: sqlite3.Connection) -> None:
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY,
            kind TEXT NOT NULL CHECK(kind IN ('events','claims')),
            symbol TEXT NOT NULL CHECK(symbol IN ('ALL','BTC','ETH')),
            captured_at_ms INTEGER NOT NULL CHECK(captured_at_ms > 0),
            source_generated_at_ms INTEGER NOT NULL CHECK(source_generated_at_ms > 0),
            snapshot_sha256 TEXT NOT NULL CHECK(length(snapshot_sha256) = 64),
            status TEXT NOT NULL CHECK(status IN ('fresh','fresh_no_events','fresh_no_claims')),
            UNIQUE(kind,symbol,captured_at_ms)
        );
        CREATE INDEX IF NOT EXISTS captures_asof
            ON captures(kind,symbol,captured_at_ms DESC);
        CREATE TABLE IF NOT EXISTS source_polls (
            id INTEGER PRIMARY KEY,
            observed_at_ms INTEGER NOT NULL CHECK(observed_at_ms > 0),
            events_status TEXT NOT NULL,
            btc_status TEXT NOT NULL,
            eth_status TEXT NOT NULL,
            events_generated_at_ms INTEGER,
            btc_generated_at_ms INTEGER,
            eth_generated_at_ms INTEGER,
            CHECK(events_status IN ('fresh','fresh_no_events','unavailable')),
            CHECK(btc_status IN ('fresh','fresh_no_claims','unavailable')),
            CHECK(eth_status IN ('fresh','fresh_no_claims','unavailable'))
        );
        CREATE INDEX IF NOT EXISTS source_polls_asof
            ON source_polls(observed_at_ms,id);
        CREATE TABLE IF NOT EXISTS event_changes (
            capture_id INTEGER NOT NULL REFERENCES captures(id),
            event_sha256 TEXT NOT NULL CHECK(length(event_sha256) = 64),
            operation TEXT NOT NULL CHECK(operation IN ('upsert','remove')),
            source TEXT,
            category TEXT,
            source_publication_ms INTEGER,
            watcher_first_seen_ms INTEGER,
            quant_available_ms INTEGER NOT NULL,
            source_url_sha256 TEXT,
            PRIMARY KEY(capture_id,event_sha256)
        );
        CREATE TABLE IF NOT EXISTS claim_changes (
            capture_id INTEGER NOT NULL REFERENCES captures(id),
            observation_id TEXT NOT NULL CHECK(length(observation_id) = 32),
            operation TEXT NOT NULL CHECK(operation IN ('upsert','remove')),
            source_group_id TEXT,
            fact_group_sha256 TEXT,
            asserted_side TEXT,
            source_available_at_ms INTEGER,
            quant_available_at_ms INTEGER NOT NULL,
            PRIMARY KEY(capture_id,observation_id)
        );
        CREATE TRIGGER IF NOT EXISTS captures_no_update BEFORE UPDATE ON captures
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS captures_no_delete BEFORE DELETE ON captures
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS source_polls_no_update BEFORE UPDATE ON source_polls
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS source_polls_no_delete BEFORE DELETE ON source_polls
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS event_changes_no_update BEFORE UPDATE ON event_changes
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS event_changes_no_delete BEFORE DELETE ON event_changes
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS claim_changes_no_update BEFORE UPDATE ON claim_changes
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS claim_changes_no_delete BEFORE DELETE ON claim_changes
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
    """)


def _kind(item: EventShadowImport | ShadowImport) -> tuple[str, str, tuple]:
    if isinstance(item, EventShadowImport):
        return "events", "ALL", item.events
    if isinstance(item, ShadowImport):
        return "claims", item.symbol, item.claims
    raise TypeError("only WhaleWatcher shadow imports are accepted")


def _incoming(kind: str, rows: tuple) -> dict[str, tuple]:
    if kind == "events":
        return {row.event_sha256: (
            row.source, row.category, row.source_publication_ms,
            row.watcher_first_seen_ms, row.source_url_sha256,
        ) for row in rows}
    return {row.observation_id: (
        row.source_group_id, row.fact_group_sha256,
        row.asserted_side, row.source_available_at_ms,
    ) for row in rows}


def _state(db: sqlite3.Connection, kind: str, symbol: str, *, asof_ms: int | None = None
           ) -> dict[str, tuple]:
    if kind == "events":
        table, key, columns = "event_changes", "event_sha256", (
            "source,category,source_publication_ms,watcher_first_seen_ms,source_url_sha256")
        availability = "quant_available_ms"
    else:
        table, key, columns = "claim_changes", "observation_id", (
            "source_group_id,fact_group_sha256,asserted_side,source_available_at_ms")
        availability = "quant_available_at_ms"
    condition = "AND c.captured_at_ms<=?" if asof_ms is not None else ""
    params = (kind, symbol, asof_ms) if asof_ms is not None else (kind, symbol)
    rows = db.execute(
        f"SELECT x.{key},x.operation,{columns},x.{availability},c.captured_at_ms "
        f"FROM {table} x JOIN captures c ON c.id=x.capture_id "
        f"WHERE c.kind=? AND c.symbol=? {condition} "
        "ORDER BY c.captured_at_ms,c.id", params,
    )
    state: dict[str, tuple] = {}
    for row in rows:
        identity, operation, *rest = row
        *metadata, available, captured = rest
        if not captured <= available <= captured + 300_000:
            raise ValueError("stored change has invalid Quant availability")
        if asof_ms is not None and available > asof_ms:
            continue
        if operation == "remove":
            state.pop(identity, None)
        else:
            state[identity] = (tuple(metadata), available)
    return state


def _append(db: sqlite3.Connection, item: EventShadowImport | ShadowImport) -> CaptureStatus:
    kind, symbol, rows = _kind(item)
    captured = item.captured_at_ms
    if item.status == "unavailable":
        return CaptureStatus(kind, symbol, "unavailable", 0, captured)
    allowed = {"fresh", "fresh_no_events"} if kind == "events" else {"fresh", "fresh_no_claims"}
    if (item.status not in allowed or item.snapshot_sha256 is None
            or item.snapshot_generated_at_ms is None
            or not 0 < item.snapshot_generated_at_ms <= captured
            or (item.status == "fresh") != bool(rows)
            or any((not captured <= row.quant_available_ms <= captured + 300_000)
                   if kind == "events" else row.quant_available_at_ms != captured
                   for row in rows)):
        raise ValueError("inconsistent shadow import")
    incoming = _incoming(kind, rows)
    event_availability = ({row.event_sha256: row.quant_available_ms for row in rows}
                          if kind == "events" else {})
    if len(incoming) != len(rows):
        raise ValueError("duplicate shadow identity")
    previous = db.execute(
        "SELECT captured_at_ms FROM captures WHERE kind=? AND symbol=? "
        "ORDER BY captured_at_ms DESC LIMIT 1", (kind, symbol),
    ).fetchone()
    if previous is not None and captured < previous[0]:
        raise ValueError("capture clock moved backwards")
    prior = {key: value[0] for key, value in _state(db, kind, symbol).items()}
    if prior == incoming and previous is not None:
        return CaptureStatus(kind, symbol, "unchanged", len(rows), captured)
    if previous is not None and captured == previous[0]:
        raise ValueError("different states share a capture millisecond")
    cursor = db.execute(
        "INSERT INTO captures(kind,symbol,captured_at_ms,source_generated_at_ms,"
        "snapshot_sha256,status) VALUES (?,?,?,?,?,?)",
        (kind, symbol, captured, item.snapshot_generated_at_ms,
         item.snapshot_sha256, item.status),
    )
    capture_id = cursor.lastrowid
    for identity in sorted(prior.keys() - incoming.keys()):
        if kind == "events":
            db.execute("INSERT INTO event_changes(capture_id,event_sha256,operation,quant_available_ms) "
                       "VALUES (?,?,?,?)", (capture_id, identity, "remove", captured))
        else:
            db.execute("INSERT INTO claim_changes(capture_id,observation_id,operation,quant_available_at_ms) "
                       "VALUES (?,?,?,?)", (capture_id, identity, "remove", captured))
    for identity in sorted(incoming):
        metadata = incoming[identity]
        if prior.get(identity) == metadata:
            continue
        if kind == "events":
            db.execute("INSERT INTO event_changes VALUES (?,?,?,?,?,?,?,?,?)",
                       (capture_id, identity, "upsert", *metadata[:4],
                        event_availability[identity], metadata[4]))
        else:
            db.execute("INSERT INTO claim_changes VALUES (?,?,?,?,?,?,?,?)",
                       (capture_id, identity, "upsert", *metadata[:3], metadata[3], captured))
    return CaptureStatus(kind, symbol, "captured", len(rows), captured)


def capture_current(watcher_data_dir: str | Path, journal_path: str | Path
                    ) -> tuple[CaptureStatus, ...]:
    """Read current snapshots and append only changed, redacted provenance."""
    source = Path(watcher_data_dir)
    journal = Path(journal_path)
    external = source / "external_intelligence_state" / "latest.json"
    events = source / "event_state" / "latest.json"
    if journal.resolve() in {external.resolve(), events.resolve()}:
        raise ValueError("journal cannot replace a WhaleWatcher snapshot")
    imports = (
        read_whalewatcher_event_shadow(events),
        read_whalewatcher_shadow_snapshot(external, "BTC"),
        read_whalewatcher_shadow_snapshot(external, "ETH"),
    )
    if all(item.status == "unavailable" for item in imports) and not journal.is_file():
        return tuple(CaptureStatus(
            *_kind(item)[:2], "unavailable", 0, item.captured_at_ms,
        ) for item in imports)
    if journal.is_file() and journal.stat().st_size > MAX_JOURNAL_BYTES:
        raise RuntimeError("prospective journal reached its 64 MiB storage cap")
    journal.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(journal)) as db:
        _initialize(db)
        with db:
            statuses = tuple(_append(db, item) for item in imports)
            db.execute(
                "INSERT INTO source_polls(observed_at_ms,events_status,btc_status,eth_status,"
                "events_generated_at_ms,btc_generated_at_ms,eth_generated_at_ms) "
                "VALUES (?,?,?,?,?,?,?)",
                (max(item.captured_at_ms for item in imports),
                 *(item.status for item in imports),
                 *(item.snapshot_generated_at_ms for item in imports)),
            )
            pages = db.execute("PRAGMA page_count").fetchone()[0]
            page_size = db.execute("PRAGMA page_size").fetchone()[0]
            if pages * page_size > MAX_JOURNAL_BYTES:
                raise RuntimeError("prospective journal reached its 64 MiB storage cap")
            return statuses


def poll_coverage(journal_path: str | Path, start_ms: int, end_ms: int,
                  *, stream: str = "joint") -> PollCoverage:
    """Audit actual polls for the selected stream without promoting other feeds."""
    if (type(start_ms) is not int or type(end_ms) is not int
            or not 0 < start_ms < end_ms):
        raise ValueError("poll coverage needs increasing positive UTC milliseconds")
    if stream not in ("joint", "events", "BTC", "ETH"):
        raise ValueError("unknown WhaleWatcher source stream")
    path = Path(journal_path)
    if not path.is_file():
        return PollCoverage(start_ms, end_ms, 0, 0, end_ms - start_ms, False)
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        try:
            rows = db.execute(
                "SELECT observed_at_ms,events_status,btc_status,eth_status "
                "FROM source_polls WHERE observed_at_ms>=? AND observed_at_ms<=? "
                "ORDER BY observed_at_ms,id", (start_ms, end_ms),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table: source_polls" not in str(exc):
                raise
            rows = []  # Old journals without source polls cannot prove continuity.
    times = [start_ms, *(int(row[0]) for row in rows), end_ms]
    largest_gap = max(right - left for left, right in zip(times, times[1:]))
    column = {"events": 1, "BTC": 2, "ETH": 3}.get(stream)
    unavailable = sum(("unavailable" in row[1:] if column is None
                       else row[column] == "unavailable") for row in rows)
    ready = (end_ms - start_ms >= MIN_OOS_MS and bool(rows)
             and largest_gap <= MAX_POLL_GAP_MS and unavailable == 0)
    return PollCoverage(start_ms, end_ms, len(rows), unavailable, largest_gap, ready)


def counts_asof(journal_path: str | Path, asof_ms: int) -> dict[str, int]:
    """Return active provenance counts from changes actually captured by then."""
    if type(asof_ms) is not int or asof_ms <= 0:
        raise ValueError("asof_ms must be a positive integer")
    path = Path(journal_path)
    if not path.is_file():
        return {"events": 0, "BTC_claims": 0, "ETH_claims": 0}
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        return {
            "events": len(_state(db, "events", "ALL", asof_ms=asof_ms)),
            "BTC_claims": len(_state(db, "claims", "BTC", asof_ms=asof_ms)),
            "ETH_claims": len(_state(db, "claims", "ETH", asof_ms=asof_ms)),
        }


def read_asof(journal_path: str | Path, asof_ms: int) -> ProvenanceSnapshot:
    """Read redacted event provenance visible by a past UTC millisecond."""
    if type(asof_ms) is not int or asof_ms <= 0:
        raise ValueError("asof_ms must be a positive integer")
    path = Path(journal_path)
    if not path.is_file():
        return ProvenanceSnapshot(asof_ms, (), (), ())
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        events = _state(db, "events", "ALL", asof_ms=asof_ms)
        btc = _state(db, "claims", "BTC", asof_ms=asof_ms)
        eth = _state(db, "claims", "ETH", asof_ms=asof_ms)

    def claim_rows(state):
        return tuple(JournalClaim(identity, *metadata, available)
                     for identity, (metadata, available) in sorted(state.items()))

    # The event metadata tuple ends with URL hash. Place the availability
    # timestamp before it in the typed result.
    event_result = tuple(
        JournalEvent(identity, metadata[0], metadata[1], metadata[2], metadata[3],
                     available, metadata[4])
        for identity, (metadata, available) in sorted(events.items())
    )
    return ProvenanceSnapshot(asof_ms, event_result, claim_rows(btc), claim_rows(eth))


def read_events_asof(journal_path: str | Path, asof_ms: int) -> tuple[JournalEvent, ...]:
    """Read only the event changes, independent of a missing claims schema/feed."""
    if type(asof_ms) is not int or asof_ms <= 0:
        raise ValueError("asof_ms must be a positive integer")
    path = Path(journal_path)
    if not path.is_file():
        return ()
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        events = _state(db, "events", "ALL", asof_ms=asof_ms)
    return tuple(
        JournalEvent(identity, metadata[0], metadata[1], metadata[2], metadata[3],
                     available, metadata[4])
        for identity, (metadata, available) in sorted(events.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture WhaleWatcher provenance for shadow research only")
    parser.add_argument("--watcher-data-dir", required=True, type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--macro-journal", type=Path)
    args = parser.parse_args()
    for item in capture_current(args.watcher_data_dir, args.journal):
        print(f"{item.kind}/{item.symbol}: {item.status}; items={item.count}")
    macro_path = args.macro_journal or args.journal.with_name("whalewatcher_macro.sqlite")
    macro = capture_macro_current(args.watcher_data_dir, macro_path)
    print(f"macro_schedule: {macro.status}; items={macro.active_events}; changes={macro.changed_events}")


if __name__ == "__main__":
    main()
