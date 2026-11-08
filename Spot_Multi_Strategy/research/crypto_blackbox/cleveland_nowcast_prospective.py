"""Small prospective Cleveland Fed CPI nowcast journal; no trading or backfill."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import json
import math
import sqlite3
import time as clock

import requests


URL = ("https://www.clevelandfed.org/-/media/files/webcharts/"
       "inflationnowcasting/nowcast_month.json?sc_lang=en")
SOURCE_LIMIT = 12 * 1024 * 1024
JOURNAL_LIMIT = 16 * 1024 * 1024
EASTERN = ZoneInfo("America/New_York")
FORECAST_KEYS = {"value", "tooltext"}
ACTUAL_KEYS = {"value", "tooltext", "anchorradius", "anchorbgcolor",
               "anchorbordercolor", "anchorborderthickness"}


@dataclass(frozen=True)
class Forecast:
    month: str
    source_day: str
    value: float


@dataclass(frozen=True)
class Actual:
    month: str
    release_day: str
    value: float


@dataclass(frozen=True)
class Capture:
    status: str
    observed_at_ms: int
    forecasts: int
    actuals: int
    linked_actuals: int
    execution_eligible: bool = False
    historical_oos_eligible: bool = False


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate source JSON key")
        out[key] = value
    return out


def _reject_constant(value):
    raise ValueError(f"nonfinite source JSON constant {value}")


def _month_add(day: date, offset: int) -> date:
    year, zero_month = divmod(day.year * 12 + day.month - 1 + offset, 12)
    return date(year, zero_month + 1, 1)


def _numeric(point: dict, allowed: set[str]) -> float:
    if not isinstance(point, dict) or set(point) != allowed:
        raise ValueError("forecast/actual point renderer schema changed")
    raw = point["value"]
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise ValueError("invalid CPI point value type")
    number = float(raw)
    if not math.isfinite(number) or abs(number) > 100:
        raise ValueError("nonfinite/out-of-range CPI point")
    return number


def _dated_labels(chart: dict, month: date) -> list[date]:
    labels = chart["categories"][0]["category"]
    output = []
    for point in labels:
        if point.get("vline"):
            continue
        label = point["label"]
        if (not isinstance(label, str) or len(label) != 5
                or label[2] != "/" or not (label[:2] + label[3:]).isdigit()):
            raise ValueError("invalid current CPI chart date label")
        mm, dd = int(label[:2]), int(label[3:])
        output.append(date(month.year + (mm < month.month), mm, dd))
    if output != sorted(set(output)):
        raise ValueError("CPI chart dates duplicate/out of order")
    return output


def parse_current(raw: bytes, observed_at_ms: int) -> tuple[str, tuple[Forecast, ...],
                                                              tuple[Actual, ...]]:
    """Read only the current/prior reference months; never import chart history."""
    if not raw or len(raw) > SOURCE_LIMIT:
        raise ValueError("missing/oversized Cleveland Fed response")
    if type(observed_at_ms) is not int or observed_at_ms <= 0:
        raise ValueError("invalid observation timestamp")
    now = datetime.fromtimestamp(observed_at_ms / 1000, tz=timezone.utc).astimezone(EASTERN)
    today = now.date()
    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                         parse_constant=_reject_constant)
    if not isinstance(payload, list) or not 100 <= len(payload) <= 240:
        raise ValueError("Cleveland Fed chart envelope changed")
    wanted = {_month_add(today, -1), _month_add(today, 0)}
    selected = {}
    for chart in payload:
        name = chart.get("chart", {}).get("subcaption")
        if not isinstance(name, str):
            raise ValueError("Cleveland Fed chart month missing")
        parts = name.split("-")
        if len(parts) != 2:
            raise ValueError("Cleveland Fed chart month malformed")
        month = date(int(parts[0]), int(parts[1]), 1)
        if month in wanted:
            if month in selected:
                raise ValueError("duplicate current CPI chart month")
            selected[month] = chart
    if set(selected) != wanted:
        raise ValueError("missing current/prior CPI chart")
    forecasts = []
    actuals = []
    for month in sorted(wanted):
        chart = selected[month]
        if chart["chart"].get("yaxisname") != "Month-over-month percent change":
            raise ValueError("Cleveland Fed monthly CPI measure changed")
        series = chart["dataset"]
        if (not isinstance(series, list) or len(series) != 8
                or series[0].get("seriesname") != "CPI Inflation"
                or series[4].get("seriesname") != "Actual CPI Inflation"):
            raise ValueError("Cleveland Fed headline CPI series changed")
        dates = _dated_labels(chart, month)
        forecast_data = series[0]["data"]
        actual_data = series[4]["data"]
        if len(forecast_data) != len(dates) or len(actual_data) != len(dates):
            raise ValueError("CPI chart dates and points do not align")
        populated_actuals = [i for i, p in enumerate(actual_data)
                             if p.get("value") not in (None, "", "-")]
        if len(populated_actuals) > 1:
            raise ValueError("multiple first CPI releases in monthly chart")
        if populated_actuals:
            index = populated_actuals[0]
            if dates[index] == today and now.time() >= time(8, 30):
                actuals.append(Actual(month.strftime("%Y-%m"), str(today),
                                      _numeric(actual_data[index], ACTUAL_KEYS)))
            # Never copy a pre-release forecast retrospectively after actual release.
            continue
        eligible = [i for i, p in enumerate(forecast_data)
                    if dates[i] <= today and p.get("value") not in (None, "", "-")]
        if not eligible:
            continue
        index = eligible[-1]
        if today - dates[index] > timedelta(days=4):
            continue
        forecasts.append(Forecast(month.strftime("%Y-%m"), str(dates[index]),
                                  _numeric(forecast_data[index], FORECAST_KEYS)))
    if not forecasts and not actuals:
        raise ValueError("no fresh current CPI data")
    return sha256(raw).hexdigest(), tuple(forecasts), tuple(actuals)


def _initialize(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS polls(
            id INTEGER PRIMARY KEY,
            observed_at_ms INTEGER NOT NULL CHECK(observed_at_ms>0),
            eastern_day TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('fresh','unavailable')),
            source_sha256 TEXT,
            CHECK((status='fresh' AND length(source_sha256)=64)
                  OR (status='unavailable' AND source_sha256 IS NULL))
        );
        CREATE INDEX IF NOT EXISTS polls_day ON polls(eastern_day,id);
        CREATE TABLE IF NOT EXISTS forecasts(
            id INTEGER PRIMARY KEY,
            poll_id INTEGER NOT NULL REFERENCES polls(id),
            month TEXT NOT NULL,
            source_day TEXT NOT NULL,
            value REAL NOT NULL,
            available_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS forecasts_asof ON forecasts(month,available_at_ms,id);
        CREATE TABLE IF NOT EXISTS actuals(
            id INTEGER PRIMARY KEY,
            poll_id INTEGER NOT NULL REFERENCES polls(id),
            month TEXT NOT NULL UNIQUE,
            release_day TEXT NOT NULL,
            value REAL NOT NULL,
            prior_forecast_id INTEGER REFERENCES forecasts(id),
            available_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS actuals_asof ON actuals(available_at_ms,id);
        CREATE TRIGGER IF NOT EXISTS polls_no_update BEFORE UPDATE ON polls
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS polls_no_delete BEFORE DELETE ON polls
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS forecasts_no_update BEFORE UPDATE ON forecasts
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS forecasts_no_delete BEFORE DELETE ON forecasts
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS actuals_no_update BEFORE UPDATE ON actuals
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS actuals_no_delete BEFORE DELETE ON actuals
            BEGIN SELECT RAISE(ABORT,'append-only'); END;
    """)


def _day(observed_at_ms: int) -> str:
    return datetime.fromtimestamp(observed_at_ms / 1000, tz=timezone.utc).astimezone(EASTERN).date().isoformat()


def _already_fresh(path: Path, eastern_day: str) -> bool:
    if not path.is_file():
        return False
    with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as db:
        row = db.execute("SELECT 1 FROM polls WHERE eastern_day=? AND status='fresh' LIMIT 1",
                         (eastern_day,)).fetchone()
        return row is not None


def capture_bytes(path: str | Path, raw: bytes | None, observed_at_ms: int) -> Capture:
    """Append a real observation, never assign the chart date as availability."""
    journal = Path(path)
    eastern_day = _day(observed_at_ms)
    if journal.is_file() and journal.stat().st_size > JOURNAL_LIMIT:
        raise RuntimeError("Cleveland nowcast journal reached its 16 MiB cap")
    if _already_fresh(journal, eastern_day):
        return Capture("already_captured_today", observed_at_ms, 0, 0, 0)
    try:
        if raw is None:
            raise ValueError("source request unavailable")
        digest, forecasts, actuals = parse_current(raw, observed_at_ms)
        status = "fresh"
    except (ValueError, TypeError, KeyError, IndexError, UnicodeError, OverflowError):
        digest, forecasts, actuals = None, (), ()
        status = "unavailable"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(journal) as db:
        db.execute("PRAGMA foreign_keys=ON")
        _initialize(db)
        linked = 0
        with db:
            cursor = db.execute(
                "INSERT INTO polls(observed_at_ms,eastern_day,status,source_sha256) VALUES(?,?,?,?)",
                (observed_at_ms, eastern_day, status, digest))
            poll_id = cursor.lastrowid
            for item in forecasts:
                db.execute(
                    "INSERT INTO forecasts(poll_id,month,source_day,value,available_at_ms) VALUES(?,?,?,?,?)",
                    (poll_id, item.month, item.source_day, item.value, observed_at_ms))
            for item in actuals:
                exists = db.execute("SELECT 1 FROM actuals WHERE month=?", (item.month,)).fetchone()
                if exists:
                    continue
                release_at = datetime.combine(date.fromisoformat(item.release_day),
                                              time(8, 30), tzinfo=EASTERN)
                release_ms = int(release_at.timestamp() * 1000)
                prior = db.execute(
                    "SELECT id FROM forecasts WHERE month=? AND available_at_ms<? "
                    "ORDER BY available_at_ms DESC,id DESC LIMIT 1",
                    (item.month, release_ms)).fetchone()
                prior_id = prior[0] if prior else None
                db.execute(
                    "INSERT INTO actuals(poll_id,month,release_day,value,prior_forecast_id,available_at_ms) "
                    "VALUES(?,?,?,?,?,?)",
                    (poll_id, item.month, item.release_day, item.value, prior_id,
                     observed_at_ms))
                linked += int(prior_id is not None)
    return Capture(status, observed_at_ms, len(forecasts), len(actuals), linked)


def _fetch() -> bytes:
    response = requests.get(URL, headers={"User-Agent": "MiniMedallionProspective/1.0"},
                            timeout=30, stream=True)
    response.raise_for_status()
    parts, size = [], 0
    try:
        for chunk in response.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > SOURCE_LIMIT:
                raise ValueError("Cleveland Fed response exceeded 12 MiB")
            parts.append(chunk)
    finally:
        response.close()
    return b"".join(parts)


def capture_current(path: str | Path, *, force: bool = False) -> Capture:
    now = clock.time_ns() // 1_000_000
    eastern = datetime.fromtimestamp(now / 1000, tz=timezone.utc).astimezone(EASTERN)
    if not force and (eastern.weekday() >= 5 or eastern.hour < 11):
        return Capture("outside_daily_window", now, 0, 0, 0)
    if _already_fresh(Path(path), eastern.date().isoformat()):
        return Capture("already_captured_today", now, 0, 0, 0)
    try:
        raw = _fetch()
    except (requests.RequestException, ValueError, OSError):
        raw = None
    return capture_bytes(path, raw, clock.time_ns() // 1_000_000)


def read_asof(path: str | Path, asof_ms: int) -> dict:
    """Return only values the collector had actually read by this time."""
    if type(asof_ms) is not int or asof_ms <= 0:
        raise ValueError("invalid as-of timestamp")
    journal = Path(path)
    if not journal.is_file():
        return {"forecasts": [], "actuals": []}
    with sqlite3.connect(f"file:{journal.resolve().as_posix()}?mode=ro", uri=True) as db:
        forecasts = [
            {"month": m, "source_day": d, "value": v, "available_at_ms": a}
            for m, d, v, a in db.execute(
                "SELECT month,source_day,value,available_at_ms FROM forecasts "
                "WHERE available_at_ms<=? ORDER BY available_at_ms,id", (asof_ms,))]
        actuals = [
            {"month": m, "release_day": d, "value": v,
             "prior_forecast_id": f, "available_at_ms": a}
            for m, d, v, f, a in db.execute(
                "SELECT month,release_day,value,prior_forecast_id,available_at_ms "
                "FROM actuals WHERE available_at_ms<=? ORDER BY available_at_ms,id",
                (asof_ms,))]
    return {"forecasts": forecasts, "actuals": actuals}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prospective Cleveland CPI nowcast capture only")
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--force", action="store_true",
                        help="allow an initial capture outside the scheduled weekday window")
    args = parser.parse_args()
    print(capture_current(args.journal, force=args.force))


if __name__ == "__main__":
    main()
