"""Read-only readiness audit for the two prospective WhaleWatcher journals.

This measures *source* continuity only. It never labels Alpha, computes PnL,
reads article prose, changes the append-only journals or authorizes trading.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sqlite3
import time

from .whalewatcher_prospective_journal import MAX_POLL_GAP_MS, MIN_OOS_MS


@dataclass(frozen=True)
class StreamReadiness:
    name: str
    state: str
    polls_total: int
    latest_continuous_polls: int
    latest_continuous_start_ms: int | None
    latest_poll_ms: int | None
    latest_poll_age_ms: int | None
    breaks_total: int
    current_span_ms: int
    five_year_continuous_source: bool
    execution_eligible: bool = False


@dataclass(frozen=True)
class JointReadiness:
    news_events: StreamReadiness
    events_claims: StreamReadiness
    macro_schedule: StreamReadiness
    joint_start_ms: int | None
    joint_span_ms: int
    five_year_continuous_source: bool
    news_macro_start_ms: int | None = None
    news_macro_span_ms: int = 0
    news_macro_five_year_continuous_source: bool = False
    independent_trading_oos_proven: bool = False
    execution_eligible: bool = False


def _read_rows(path: Path, query: str) -> list[tuple] | None:
    if not path.is_file():
        return None
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as db:
            return db.execute(query).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"prospective journal invalid: {path.name}: {exc}") from exc


def _five_calendar_years_elapsed(start_ms: int, now_ms: int) -> bool:
    start = datetime.fromtimestamp(start_ms / 1_000, tz=timezone.utc)
    end = datetime.fromtimestamp(now_ms / 1_000, tz=timezone.utc)
    try:
        fifth_anniversary = start.replace(year=start.year + 5)
    except ValueError:  # 29 February -> 1 March in a non-leap fifth year.
        fifth_anniversary = start.replace(year=start.year + 5, month=3, day=1)
    return end >= fifth_anniversary


def _stream(name: str, rows: list[tuple] | None, now_ms: int) -> StreamReadiness:
    if rows is None:
        return StreamReadiness(name, "missing", 0, 0, None, None, None, 0, 0, False)
    start = last_valid = last_seen = None
    continuous_polls = breaks = 0
    for item in rows:
        stamp = item[0]
        if type(stamp) is not int or stamp <= 0 or stamp > now_ms:
            raise ValueError(f"{name}: invalid or future poll timestamp")
        if last_seen is not None and stamp < last_seen:
            raise ValueError(f"{name}: poll timestamps out of order")
        if last_seen is not None and stamp - last_seen > MAX_POLL_GAP_MS:
            breaks += 1
            start = last_valid = None
            continuous_polls = 0
        last_seen = stamp
        if name == "events_claims":
            valid = all(status in ("fresh", "fresh_no_events", "fresh_no_claims")
                        for status in item[1:])
        elif name == "news_events":
            valid = item[1] in ("fresh", "fresh_no_events")
        else:
            valid = item[1] == "fresh"
        if not valid:
            breaks += 1
            start = last_valid = None
            continuous_polls = 0
            continue
        if start is None:
            start = stamp
        last_valid = stamp
        continuous_polls += 1
    if last_seen is None:
        return StreamReadiness(name, "empty", 0, 0, None, None, None, 0, 0, False)
    age = now_ms - last_seen
    if age > MAX_POLL_GAP_MS:
        breaks += 1
        state = "stale"
        span = 0
        start = None
        continuous_polls = 0
    elif start is None:
        state = "unavailable"
        span = 0
    else:
        state = "fresh"
        span = now_ms - start
    ready = (state == "fresh" and continuous_polls > 0 and start is not None
             and span >= MIN_OOS_MS and _five_calendar_years_elapsed(start, now_ms))
    return StreamReadiness(name, state, len(rows), continuous_polls,
                           start, last_seen, age, breaks, span, ready)


def assess(journal_dir: str | Path, now_ms: int | None = None) -> JointReadiness:
    """Find the latest uninterrupted, currently live common source interval."""
    if now_ms is None:
        now_ms = time.time_ns() // 1_000_000
    if type(now_ms) is not int or now_ms <= 0:
        raise ValueError("now_ms must be a positive UTC millisecond integer")
    directory = Path(journal_dir)
    source_rows = _read_rows(
        directory / "whalewatcher_shadow.sqlite",
        "SELECT observed_at_ms,events_status,btc_status,eth_status "
        "FROM source_polls ORDER BY observed_at_ms,id")
    news = _stream("news_events",
                   [(row[0], row[1]) for row in source_rows]
                   if source_rows is not None else None, now_ms)
    events = _stream("events_claims", source_rows, now_ms)
    macro = _stream("macro_schedule", _read_rows(
        directory / "whalewatcher_macro.sqlite",
        "SELECT observed_at_ms,status FROM polls ORDER BY observed_at_ms,id"), now_ms)
    if events.latest_continuous_start_ms is None or macro.latest_continuous_start_ms is None:
        joint_start = None
        joint_span = 0
    else:
        joint_start = max(events.latest_continuous_start_ms,
                          macro.latest_continuous_start_ms)
        joint_span = now_ms - joint_start
    joint_ready = (events.five_year_continuous_source
                   and macro.five_year_continuous_source
                   and joint_span >= MIN_OOS_MS
                   and joint_start is not None
                   and _five_calendar_years_elapsed(joint_start, now_ms))
    if news.latest_continuous_start_ms is None or macro.latest_continuous_start_ms is None:
        news_macro_start = None
        news_macro_span = 0
    else:
        news_macro_start = max(news.latest_continuous_start_ms,
                               macro.latest_continuous_start_ms)
        news_macro_span = now_ms - news_macro_start
    news_macro_ready = (news.five_year_continuous_source
                        and macro.five_year_continuous_source
                        and news_macro_start is not None
                        and news_macro_span >= MIN_OOS_MS
                        and _five_calendar_years_elapsed(news_macro_start, now_ms))
    return JointReadiness(news, events, macro, joint_start, joint_span, joint_ready,
                          news_macro_start, news_macro_span, news_macro_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit WhaleWatcher source continuity only")
    parser.add_argument("journal_dir", type=Path,
                        help="directory containing the two prospective SQLite journals")
    args = parser.parse_args()
    print(json.dumps(asdict(assess(args.journal_dir)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
