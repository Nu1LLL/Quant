"""V2 structural reader for dated Federal Reserve H.4.1 release pages.

Only the preregistered page-layout correction differs from v1: the target
table's two preceding element siblings identify its heading and unit, and
the publication date comes from the page title.  The index, transport,
numeric column, and point-in-time checks remain unchanged.  No raw page or
numeric series is persisted by this module.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import math
import re
from typing import Callable

import pandas as pd

from research.fed_reserves_vintage_data import (
    EXPECTED_RELEASES,
    RELEASE_BASE,
    RELEASE_INDEX_URL,
    ROW_LABEL,
    UNIT,
    ReserveBundle,
    ReserveObservation,
    VintageDataError,
    _Element,
    _LONG_DATE,
    _Payload,
    _Tree,
    _direct_cells,
    _fetch,
    _http_fetch,
    _normal,
    _parse_long_date,
    _parse_release_index,
    _release_url,
)


TABLE_HEADING = "1. Factors Affecting Reserve Balances of Depository Institutions (continued)"
PARSER_VERSION = "h41-adjacent-p-title-v2"


def _parse_release_page(body: bytes, release_date: pd.Timestamp) -> tuple[ReserveObservation, dict]:
    """Extract the sole Wednesday level after exact, local page checks."""
    try:
        html = body.decode("utf-8-sig", errors="strict")
    except UnicodeError as exc:
        raise VintageDataError("H.4.1 HTML is not valid UTF-8") from exc
    tree = _Tree()
    tree.feed(html)
    tree.close()

    # Some official pages have an unrelated <title>Lock</title>.  It must
    # neither supply a release date nor invalidate one unique H.4.1 title.
    titles = [
        _normal(node.text())
        for node in tree.root.descendants("title")
        if re.search(r"\bH\.4\.1\b", _normal(node.text()), re.I)
        and _LONG_DATE.search(_normal(node.text()))
    ]
    if len(titles) != 1:
        raise VintageDataError("expected one dated H.4.1 page title")
    if _parse_long_date(titles[0]) != release_date:
        raise VintageDataError("H.4.1 title date differs from release index/URL")

    targets: list[tuple[_Element, list[_Element]]] = []
    for table in tree.root.descendants("table"):
        for row in table.descendants("tr"):
            cells = _direct_cells(row)
            if cells and _normal(cells[0].text()) == ROW_LABEL:
                targets.append((table, cells))
    if len(targets) != 1:
        raise VintageDataError(f"expected one exact {ROW_LABEL!r} Table 1 row, got {len(targets)}")
    table, cells = targets[0]
    if len(cells) != 5:
        raise VintageDataError("reserve row must contain exactly five cells")

    if table.parent is None:
        raise VintageDataError("reserve table has no parent element")
    # _Element is a dataclass, so list.index would compare recursive DOM trees
    # for equality.  The relationship here is object identity instead.
    table_position = next(
        index for index, child in enumerate(table.parent.children) if child is table
    )
    siblings = [
        child for child in table.parent.children[:table_position]
        if isinstance(child, _Element)
    ]
    if (
        len(siblings) < 2
        or siblings[-2].tag != "p"
        or siblings[-1].tag != "p"
        or _normal(siblings[-2].text()) != TABLE_HEADING
        or _normal(siblings[-1].text()) != UNIT
    ):
        raise VintageDataError("reserve row is not in exact adjacent Table 1 heading/unit context")

    last = cells[-1]
    # Either template's spelling is valid, but two simultaneous attributes or
    # repeated refs make the last-column identity ambiguous.
    header_attr = last.attrs.get("header", "").strip()
    headers_attr = last.attrs.get("headers", "").strip()
    if bool(header_attr) == bool(headers_attr):
        raise VintageDataError("last value cell must use exactly one header(s) attribute")
    refs = (header_attr or headers_attr).split()
    if not refs or len(refs) != len(set(refs)):
        raise VintageDataError("last value cell has missing or repeated header IDs")
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
    return ReserveObservation(release_date, observed_date, value), {
        "label": ROW_LABEL,
        "unit": UNIT,
        "table_heading": TABLE_HEADING,
        "header_ids": refs,
        "wednesday_header": wednesday[0],
        "row_cell_count": len(cells),
        "parser_version": PARSER_VERSION,
    }


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
    """Fetch all frozen release dates with strict PIT and provenance checks."""
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
        "parser_version": PARSER_VERSION,
        "pages": {key: pages[key] for key in sorted(pages)},
    }
    return ReserveBundle(observations, manifest)
