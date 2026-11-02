"""Strict point-in-time reader for official Federal Reserve H.4.1 releases.

The only numeric series accepted is the *Wednesday level* in Table 1's
``Reserve balances with Federal Reserve Banks`` row.  Each value remains
attached to its actual H.4.1 release date; an observation date is never a
trading availability date.  Nothing is written to disk.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
import re
import time
from typing import Callable

import pandas as pd
import requests


RELEASE_INDEX_URL = "https://www.federalreserve.gov/releases/h41/releaseDates.json"
RELEASE_BASE = "https://www.federalreserve.gov/releases/h41"
FIRST_RELEASE = pd.Timestamp("2020-11-01", tz="UTC")
LAST_RELEASE = pd.Timestamp("2025-12-31", tz="UTC")
EXPECTED_RELEASES = 269
ROW_LABEL = "Reserve balances with Federal Reserve Banks"
UNIT = "Millions of dollars"
TABLE_TITLE = "Table 1. Factors Affecting Reserve Balances"
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|"
    r"Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
)
_LONG_DATE = re.compile(rf"\b({_MONTH})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I)
_SPACE = re.compile(r"\s+")
_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input",
                   "link", "meta", "param", "source", "track", "wbr"})


class VintageDataError(RuntimeError):
    """Official release vintage cannot be used without guessing or repair."""


@dataclass(frozen=True)
class ReserveObservation:
    release_date: pd.Timestamp
    observed_date: pd.Timestamp
    value: float


@dataclass(frozen=True)
class ReserveBundle:
    observations: tuple[ReserveObservation, ...]
    manifest: dict


@dataclass(frozen=True)
class _Payload:
    body: bytes
    status_code: int = 200
    attempts: int = 1


@dataclass
class _Element:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["_Element | str"] = field(default_factory=list)
    parent: "_Element | None" = None
    before_text: str = ""

    def text(self) -> str:
        return " ".join(
            child.text() if isinstance(child, _Element) else child
            for child in self.children
        )

    def descendants(self, tag: str) -> list["_Element"]:
        result = []
        for child in self.children:
            if isinstance(child, _Element):
                if child.tag == tag:
                    result.append(child)
                result.extend(child.descendants(tag))
        return result


class _Tree(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Element("document")
        self.stack = [self.root]
        self.text_seen = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Element(
            tag.lower(), {name.lower(): value or "" for name, value in attrs},
            parent=self.stack[-1],
            before_text=self.text_seen[-2400:] if tag.lower() == "table" else "",
        )
        self.stack[-1].children.append(node)
        if tag.lower() not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag.lower():
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)
        self.text_seen += " " + data


def _normal(text: str) -> str:
    return _SPACE.sub(" ", text.replace("\xa0", " ")).strip()


def _http_fetch(url: str) -> _Payload:
    # Fixed transport retry only: never substitute a different date, page,
    # series, or source.  404 and other non-transient statuses fail immediately.
    for attempt in range(1, 5):
        try:
            response = requests.get(url, timeout=(10, 45))
        except requests.RequestException as exc:
            if attempt == 4:
                raise VintageDataError(
                    f"H.4.1 request failed after {attempt} attempts: {url}"
                ) from exc
        else:
            if response.status_code != 429 and not 500 <= response.status_code <= 599:
                return _Payload(response.content, response.status_code, attempt)
            if attempt == 4:
                return _Payload(response.content, response.status_code, attempt)
        time.sleep((0.5, 1.0, 2.0)[attempt - 1])
    raise AssertionError("unreachable H.4.1 retry loop")


def _fetch(fetcher: Callable[[str], bytes | _Payload], url: str) -> _Payload:
    try:
        item = fetcher(url)
    except VintageDataError:
        raise
    except Exception as exc:
        raise VintageDataError(f"H.4.1 request failed: {url}") from exc
    if isinstance(item, bytes):
        item = _Payload(item)
    if not isinstance(item, _Payload) or not isinstance(item.body, bytes):
        raise VintageDataError(f"unexpected H.4.1 response type: {url}")
    if item.status_code != 200 or not item.body:
        raise VintageDataError(f"H.4.1 HTTP {item.status_code} or empty body: {url}")
    return item


def _parse_release_index(body: bytes) -> tuple[pd.Timestamp, ...]:
    """Read the official date index, including holiday-delayed releases."""
    try:
        years = json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise VintageDataError("invalid official releaseDates.json") from exc
    if not isinstance(years, list):
        raise VintageDataError("releaseDates.json is not a year list")
    dates: list[pd.Timestamp] = []
    months_seen: set[str] = set()
    for year in years:
        if not isinstance(year, dict) or not re.fullmatch(r"\d{4}", str(year.get("yearValue", ""))):
            raise VintageDataError("invalid release index year")
        year_number = int(year["yearValue"])
        months = year.get("Months")
        if not isinstance(months, list):
            raise VintageDataError("invalid release index months")
        for month in months:
            if not isinstance(month, dict):
                raise VintageDataError("invalid release index month")
            month_code = month.get("MonthValue")
            if not isinstance(month_code, str) or not re.fullmatch(r"\d{6}", month_code):
                raise VintageDataError("invalid release index month code")
            if int(month_code[:4]) != year_number or not 1 <= int(month_code[4:]) <= 12:
                raise VintageDataError("release index month/year mismatch")
            raw_dates = month.get("Dates")
            if not isinstance(raw_dates, list):
                raise VintageDataError("invalid release index date list")
            if "202011" <= month_code <= "202512":
                if month_code in months_seen:
                    raise VintageDataError("duplicate release index month")
                months_seen.add(month_code)
            for raw in raw_dates:
                if not isinstance(raw, str) or not re.fullmatch(r"\d{8}", raw):
                    raise VintageDataError("invalid release index date")
                try:
                    parsed = pd.Timestamp(datetime.strptime(raw, "%Y%m%d"), tz="UTC")
                except ValueError as exc:
                    raise VintageDataError("invalid calendar release date") from exc
                if raw[:6] != month_code:
                    raise VintageDataError("release date/month mismatch")
                if FIRST_RELEASE <= parsed <= LAST_RELEASE:
                    dates.append(parsed)
    expected_months = set(pd.period_range("2020-11", "2025-12", freq="M").strftime("%Y%m"))
    if months_seen != expected_months:
        raise VintageDataError("incomplete or extra release index month coverage")
    if len(dates) != EXPECTED_RELEASES or len(set(dates)) != EXPECTED_RELEASES:
        raise VintageDataError(
            f"expected {EXPECTED_RELEASES} unique release dates, got {len(dates)}"
        )
    for year in range(2021, 2026):
        if sum(day.year == year for day in dates) != 52:
            raise VintageDataError(f"{year}: expected exactly 52 releases")
    return tuple(sorted(dates))


def _release_url(release_date: pd.Timestamp) -> str:
    if release_date.tzinfo is None or release_date.utcoffset() != pd.Timedelta(0):
        raise ValueError("release_date must be UTC-aware")
    return f"{RELEASE_BASE}/{release_date:%Y%m%d}/h41.htm"


def _parse_long_date(text: str) -> pd.Timestamp:
    matches = list(_LONG_DATE.finditer(text))
    if len(matches) != 1:
        raise VintageDataError("expected exactly one full date in referenced header")
    match = matches[0]
    month, day, year = match.groups()
    month = month.rstrip(".")
    formats = ("%B %d %Y", "%b %d %Y")
    for fmt in formats:
        try:
            return pd.Timestamp(datetime.strptime(f"{month} {day} {year}", fmt), tz="UTC")
        except ValueError:
            pass
    raise VintageDataError("invalid full date in referenced header")


def _direct_cells(row: _Element) -> list[_Element]:
    return [child for child in row.children
            if isinstance(child, _Element) and child.tag in {"th", "td"}]


def _parse_release_page(body: bytes, release_date: pd.Timestamp) -> tuple[ReserveObservation, dict]:
    try:
        html = body.decode("utf-8-sig", errors="strict")
    except UnicodeError as exc:
        raise VintageDataError("H.4.1 HTML is not valid UTF-8") from exc
    tree = _Tree()
    tree.feed(html)
    tree.close()
    full_text = _normal(tree.root.text())
    tables = tree.root.descendants("table")
    targets: list[tuple[_Element, _Element, list[_Element]]] = []
    for table in tables:
        for row in table.descendants("tr"):
            cells = _direct_cells(row)
            if cells and _normal(cells[0].text()) == ROW_LABEL:
                targets.append((table, row, cells))
    if len(targets) != 1:
        raise VintageDataError(f"expected one exact {ROW_LABEL!r} Table 1 row, got {len(targets)}")
    table, _, cells = targets[0]
    if len(cells) != 5:
        raise VintageDataError("reserve row must contain exactly five cells")
    before = _normal(table.before_text)
    context = before[-1500:] + " " + _normal(table.text())[:1000]
    if not re.search(r"\bTable\s*1\s*\.\s*Factors Affecting Reserve Balances\b", context, re.I):
        raise VintageDataError("reserve row is not in exact Table 1 context")
    if UNIT.lower() not in context.lower():
        raise VintageDataError("H.4.1 Table 1 unit mismatch")
    # The published title/release stamp occurs before Table 1.  Requiring an
    # explicit date in that portion prevents a later page being mislabelled
    # with a stale archive URL.  Observation-header dates are after Table 1.
    release_stamp = release_date.strftime("%B %d, %Y")
    before_dates = [_parse_long_date(m.group()) for m in _LONG_DATE.finditer(before)]
    if release_date not in before_dates:
        raise VintageDataError(
            f"H.4.1 page publication date does not match {release_stamp}"
        )
    last = cells[-1]
    refs = (last.attrs.get("headers") or last.attrs.get("header") or "").split()
    if not refs:
        raise VintageDataError("last value cell lacks a referenced Wednesday header")
    id_map: dict[str, _Element] = {}
    for node in [table, *table.descendants("th"), *table.descendants("td")]:
        node_id = node.attrs.get("id")
        if node_id:
            if node_id in id_map:
                raise VintageDataError("duplicate Table 1 header id")
            id_map[node_id] = node
    referenced = []
    for ref in refs:
        if ref not in id_map:
            raise VintageDataError(f"unresolved last-column header id: {ref}")
        referenced.append(_normal(id_map[ref].text()))
    wednesday = [header for header in referenced if re.search(r"\bWednesday\b", header, re.I)]
    if len(wednesday) != 1:
        raise VintageDataError("last-column header is not uniquely Wednesday")
    observed_date = _parse_long_date(wednesday[0])
    if observed_date.day_name() != "Wednesday" or observed_date >= release_date:
        raise VintageDataError("Wednesday observation date invalid for release")
    raw_value = _normal(last.text())
    if not re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", raw_value):
        raise VintageDataError("reserve value is not an unambiguous nonnegative number")
    value = float(raw_value.replace(",", ""))
    if not math.isfinite(value) or value < 0:
        raise VintageDataError("reserve value is nonfinite or negative")
    metadata = {
        "label": ROW_LABEL,
        "unit": UNIT,
        "header_ids": refs,
        "wednesday_header": wednesday[0],
        "row_cell_count": len(cells),
    }
    return ReserveObservation(release_date, observed_date, value), metadata


def _load_page(
    release_date: pd.Timestamp, fetcher: Callable[[str], bytes | _Payload]
) -> tuple[ReserveObservation, dict]:
    url = _release_url(release_date)
    response = _fetch(fetcher, url)
    observation, parsed = _parse_release_page(response.body, release_date)
    return observation, {
        "url": url,
        "release_date": release_date.date().isoformat(),
        "observed_date": observation.observed_date.date().isoformat(),
        "http_status": response.status_code,
        "http_attempts": response.attempts,
        "sha256": sha256(response.body).hexdigest(),
        "bytes": len(response.body),
        **parsed,
    }


def load_and_audit(
    *, fetcher: Callable[[str], bytes | _Payload] | None = None,
    max_workers: int = 4,
) -> ReserveBundle:
    """Fetch 269 official as-released pages and verify exact PIT provenance.

    ``fetcher`` is injectable for fixture-only tests and must return raw bytes
    or ``_Payload``.  Raw pages and the values themselves are never persisted.
    """
    if not 1 <= max_workers <= 4:
        raise ValueError("max_workers must be 1..4")
    fetch = fetcher or _http_fetch
    index = _fetch(fetch, RELEASE_INDEX_URL)
    release_dates = _parse_release_index(index.body)
    by_day: dict[pd.Timestamp, ReserveObservation] = {}
    pages: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_load_page, day, fetch): day for day in release_dates}
        for future in as_completed(futures):
            day = futures[future]
            try:
                observation, page_metadata = future.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
            by_day[day] = observation
            pages[day.date().isoformat()] = page_metadata
    observations = tuple(by_day[day] for day in release_dates)
    if len({item.observed_date for item in observations}) != EXPECTED_RELEASES:
        raise VintageDataError("duplicate Wednesday observation dates")
    if any(a.observed_date >= b.observed_date for a, b in zip(observations, observations[1:])):
        raise VintageDataError("Wednesday observation dates are not strictly increasing")
    manifest = {
        "release_index": {
            "url": RELEASE_INDEX_URL,
            "http_status": index.status_code,
            "http_attempts": index.attempts,
            "sha256": sha256(index.body).hexdigest(),
            "bytes": len(index.body),
        },
        "release_count": len(observations),
        "first_release": release_dates[0].date().isoformat(),
        "last_release": release_dates[-1].date().isoformat(),
        "source": RELEASE_BASE,
        "row_label": ROW_LABEL,
        "unit": UNIT,
        "pages": {key: pages[key] for key in sorted(pages)},
    }
    return ReserveBundle(observations, manifest)
