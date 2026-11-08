"""Frozen WhaleWatcher CPI release bridge, current-vintage CPI and BTC spot source."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import csv
import io
import math
import re

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure, _fetch, _load_month, _manifest


ROOT = Path(__file__).resolve().parent
EVENT_MANIFEST = ROOT / "preregistrations/2026-11-08_whalewatcher_cpi_release_manifest.csv"
EVENT_SHA = "075ed48d62d585e1f3f4f2f798c9cfea697517bbae3d55a075f3cefa8f7a5fc7"
CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"
CPI_SHA = "cfab9b9be4a5d4f06253c4dcc3f7e0b28b1d58837e710dd7cb1fd54a49e7ad7c"
EVENT_COLUMNS = ("release_utc", "reference_month", "bls_url", "timestamp_basis",
                 "snapshot_verified_at_ms")
FIRST_CPI = date(1913, 1, 1)
LAST_CPI = date(2026, 9, 1)
PRICE_FIRST = date(2020, 12, 1)
PRICE_LAST = date(2025, 12, 31)


def release_manifest(body: bytes) -> tuple[dict, ...]:
    if sha256(body).hexdigest() != EVENT_SHA:
        raise SourceFailure("WhaleWatcher CPI release manifest SHA changed")
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
        if tuple(reader.fieldnames or ()) != EVENT_COLUMNS:
            raise SourceFailure("WhaleWatcher CPI release manifest schema changed")
        raw = list(reader)
        if len(raw) != 71:
            raise SourceFailure("WhaleWatcher CPI release count changed")
        out: list[dict] = []
        for row in raw:
            if set(row) != set(EVENT_COLUMNS):
                raise SourceFailure("WhaleWatcher CPI release row schema changed")
            ts = datetime.fromisoformat(row["release_utc"].replace("Z", "+00:00"))
            if ts.tzinfo != timezone.utc or (ts.hour, ts.minute, ts.second) not in ((12, 30, 0), (13, 30, 0)):
                raise SourceFailure("WhaleWatcher CPI release UTC time invalid")
            if not 2020 <= ts.year <= 2025:
                raise SourceFailure("WhaleWatcher CPI release year invalid")
            month = (ts.date().replace(day=1) - timedelta(days=1)).replace(day=1)
            if row["reference_month"] != month.strftime("%Y-%m"):
                raise SourceFailure("WhaleWatcher CPI reference month changed")
            if (row["bls_url"] != "https://www.bls.gov/news.release/archives/"
                    f"cpi_{ts:%m%d%Y}.htm"):
                raise SourceFailure("WhaleWatcher CPI BLS URL/date changed")
            if (row["timestamp_basis"] != "official_archive_embargo_header_snapshot"
                    or not re.fullmatch(r"[0-9]{13}", row["snapshot_verified_at_ms"])):
                raise SourceFailure("WhaleWatcher CPI timestamp evidence invalid")
            out.append({"released_at": ts, "reference_month": month,
                        "bls_url": row["bls_url"], "snapshot_verified_at_ms": int(row["snapshot_verified_at_ms"])})
        if [x["released_at"] for x in out] != sorted({x["released_at"] for x in out}):
            raise SourceFailure("WhaleWatcher CPI release timestamps duplicate/unsorted")
        counts = Counter(x["released_at"].year for x in out)
        if counts != {**{y: 12 for y in range(2020, 2025)}, 2025: 11}:
            raise SourceFailure("WhaleWatcher CPI annual release coverage changed")
        return tuple(out)
    except SourceFailure:
        raise
    except (UnicodeError, csv.Error, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure(f"WhaleWatcher CPI release manifest invalid: {exc}") from exc


def parse_cpi(body: bytes) -> dict[date, float | None]:
    if sha256(body).hexdigest() != CPI_SHA:
        raise SourceFailure("current-vintage FRED CPI response SHA changed")
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
        if tuple(reader.fieldnames or ()) != ("observation_date", "CPIAUCNS"):
            raise SourceFailure("FRED CPI CSV header changed")
        raw = list(reader)
        expected = pd.period_range("1913-01", "2026-09", freq="M")
        if len(raw) != len(expected) or len(raw) != 1365:
            raise SourceFailure("FRED CPI month count changed")
        out: dict[date, float | None] = {}
        for row, period in zip(raw, expected):
            month = period.to_timestamp().date()
            if set(row) != {"observation_date", "CPIAUCNS"} or row["observation_date"] != month.isoformat():
                raise SourceFailure(f"FRED CPI date/schema invalid at {month}")
            value = row["CPIAUCNS"]
            if value == ".":
                out[month] = None
            elif re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value or ""):
                number = float(value)
                if not math.isfinite(number) or number <= 0:
                    raise SourceFailure(f"FRED CPI nonpositive at {month}")
                out[month] = number
            else:
                raise SourceFailure(f"FRED CPI numeric schema invalid at {month}")
        required = pd.period_range("2019-01", "2025-11", freq="M")
        missing = {p.to_timestamp().date() for p in required if out[p.to_timestamp().date()] is None}
        if missing != {date(2025, 10, 1)}:
            raise SourceFailure(f"FRED CPI 2019–2025 missing-value mask changed: {missing}")
        return out
    except SourceFailure:
        raise
    except (UnicodeError, csv.Error, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure(f"FRED CPI CSV invalid: {exc}") from exc


def load_sources() -> tuple[tuple[dict, ...], dict[date, float | None], pd.DataFrame, float, dict]:
    manifest_bytes = EVENT_MANIFEST.read_bytes()
    events = release_manifest(manifest_bytes)
    cpi_bytes = _fetch(CPI_URL)
    cpi = parse_cpi(cpi_bytes)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_load_month, _manifest()))
    parts.sort(key=lambda x: x[0])
    terminal_open = parts[-1][1]
    if not isinstance(terminal_open, float) or not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("terminal BTC spot open absent")
    frames = [x[1] for x in parts[:-1]]
    if not all(isinstance(x, pd.DataFrame) for x in frames):
        raise SourceFailure("nonterminal BTC spot frame invalid")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(PRICE_FIRST, PRICE_LAST, tz="UTC", freq="D")):
        raise SourceFailure("joined BTC spot date grid changed")
    audit = {"whalewatcher_manifest": {"sha256": EVENT_SHA, "release_count": len(events),
                                       "formal_release_count": sum(2021 <= x["released_at"].year <= 2025 for x in events),
                                       "source_snapshot_sha256": "2241740284252a8d22447bf266f649729a7ca5e3e9169ab55b3449cc853f19ee"},
             "fred_cpi": {"url": CPI_URL, "sha256": sha256(cpi_bytes).hexdigest(),
                          "bytes": len(cpi_bytes), "months": len(cpi),
                          "current_vintage_not_point_in_time": True},
             "binance_spot": [x[2] for x in parts]}
    return events, cpi, prices, terminal_open, audit
