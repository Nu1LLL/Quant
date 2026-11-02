"""Synthetic-only tests for the preregistered H.4.1 page-parser correction."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.fed_reserves_vintage_data import (
    EXPECTED_RELEASES,
    RELEASE_INDEX_URL,
    ROW_LABEL,
    UNIT,
    VintageDataError,
    _release_url,
)
from research.fed_reserves_vintage_data_v2 import (
    PARSER_VERSION,
    TABLE_HEADING,
    _parse_release_page,
    load_and_audit,
)


def _page(
    release: str | pd.Timestamp,
    *,
    template: str = "new",
    preceding_tables: int = 2,
    title_day: pd.Timestamp | None = None,
    duplicate_title: bool = False,
    include_lock: bool = True,
    heading: str = TABLE_HEADING,
    unit: str = UNIT,
    separator: str = " \n ",
    extra_element: str = "",
    label: str = ROW_LABEL,
    duplicate_row: bool = False,
    cell_count: int = 5,
    header_day: pd.Timestamp | None = None,
    header_text: str | None = None,
    last_attr: str | None = None,
    value: str = "1,234.5",
) -> bytes:
    """Mimic old/new DOM topology with deliberately fictitious balances."""
    release = pd.Timestamp(release)
    observed = header_day or release - pd.Timedelta(days=(release.weekday() - 2) % 7 or 7)
    title_day = title_day or release
    if template == "old":
        title = (
            "FRB: H.4.1 Release-- Factors Affecting Reserve Balances -- "
            f"{title_day:%A, %B %d, %Y}"
        )
        label_tag, label_id = "th", "t3r16"
        first_header, wed_header = "t3r16", "t3c2"
        default_attr = f"headers='{wed_header} {first_header}'"
        header_cells = f"<th id='{wed_header}'>Wednesday {observed:%b %d, %Y}</th>"
    elif template == "new":
        title = (
            "Federal Reserve Balance Sheet: Factors Affecting Reserve Balances - "
            f"H.4.1 - {title_day:%B %d, %Y}"
        )
        label_tag, label_id = "td", "t2r16c1"
        first_header, wed_header = "t2r1c1", "t2h1c3"
        default_attr = f"header='{first_header} {wed_header}'"
        header_cells = (
            f"<td id='{first_header}'>Other header</td>"
            f"<td id='{wed_header}'>Wednesday {observed:%b %d, %Y}</td>"
        )
    else:
        raise ValueError("unknown synthetic template")
    if header_text is not None:
        header_cells = header_cells.replace(
            f"Wednesday {observed:%b %d, %Y}", header_text
        )
    last_attr = default_attr if last_attr is None else last_attr
    cells = [f"<{label_tag} id='{label_id}'>{label}</{label_tag}>"]
    cells.extend("<td>0</td>" for _ in range(cell_count - 2))
    cells.append(f"<td {last_attr}>{value}</td>")
    row = "<tr>" + "".join(cells) + "</tr>"
    table = "<table><tr>" + header_cells + "</tr>" + row
    if duplicate_row:
        table += row
    table += "</table>"
    head = f"<head><title>{title}</title>"
    if duplicate_title:
        head += f"<title>{title}</title>"
    if include_lock:
        head += "<title>Lock</title>"
    head += "</head>"
    dummy_tables = "<table><tr><td>Unrelated</td></tr></table>" * preceding_tables
    local = (
        f"<p>{heading}</p>{separator}<p>{unit}</p>"
        f"{separator}{extra_element}{table}"
    )
    if template == "new":
        local = "<div>" + local + "</div>"
    return f"<html>{head}<body>{dummy_tables}{local}</body></html>".encode()


def _synthetic_index(dates: pd.DatetimeIndex) -> bytes:
    years = []
    for year in sorted(set(dates.year), reverse=True):
        months = []
        for month in sorted(set(dates[dates.year == year].month), reverse=True):
            part = dates[(dates.year == year) & (dates.month == month)]
            months.append({
                "MonthName": datetime(year, month, 1).strftime("%B"),
                "MonthValue": f"{year}{month:02d}",
                "Dates": [f"{day:%Y%m%d}" for day in part],
            })
        years.append({"yearValue": str(year), "Months": months})
    return json.dumps(years).encode()


class H41V2PageTests(unittest.TestCase):
    def test_old_new_and_holiday_templates_with_unrelated_lock_title(self):
        cases = (
            ("2020-11-05", "old", 2, "2020-11-04"),
            ("2025-12-29", "new", 3, "2025-12-24"),
        )
        for day, template, preceding_tables, observed in cases:
            with self.subTest(day=day, template=template):
                release = pd.Timestamp(day, tz="UTC")
                observation, metadata = _parse_release_page(
                    _page(release, template=template, preceding_tables=preceding_tables),
                    release,
                )
                self.assertEqual(observation.release_date, release)
                self.assertEqual(observation.observed_date, pd.Timestamp(observed, tz="UTC"))
                self.assertEqual(observation.value, 1234.5)
                self.assertEqual(metadata["table_heading"], TABLE_HEADING)
                self.assertEqual(metadata["unit"], UNIT)
                self.assertEqual(metadata["row_cell_count"], 5)
                self.assertEqual(metadata["parser_version"], PARSER_VERSION)

    def test_title_must_be_one_dated_h41_title_matching_release_index(self):
        day = pd.Timestamp("2025-12-29", tz="UTC")
        bad_cases = (
            {"title_day": pd.Timestamp("2025-12-25", tz="UTC")},
            {"duplicate_title": True},
        )
        for change in bad_cases:
            with self.subTest(change=change), self.assertRaises(VintageDataError):
                _parse_release_page(_page(day, **change), day)
        undated = _page(day).replace(b"H.4.1 - December 29, 2025", b"H.4.1 - undated")
        with self.assertRaisesRegex(VintageDataError, "dated H.4.1 page title"):
            _parse_release_page(undated, day)

    def test_heading_unit_must_be_immediate_element_siblings(self):
        day = pd.Timestamp("2025-12-29", tz="UTC")
        bad_cases = (
            {"heading": "Table 1. Factors Affecting Reserve Balances"},
            {"heading": "1. Factors Affecting Reserve Balances of Depository Institutions"},
            {"unit": "Billions of dollars"},
            {"extra_element": "<span>Unrelated</span>"},
        )
        for change in bad_cases:
            with self.subTest(change=change), self.assertRaisesRegex(VintageDataError, "adjacent"):
                _parse_release_page(_page(day, **change), day)
        # Whitespace text nodes do not interrupt element-sibling adjacency.
        _parse_release_page(_page(day, separator="\n   \t  "), day)

    def test_row_and_last_column_header_fail_closed(self):
        day = pd.Timestamp("2025-12-29", tz="UTC")
        bad_cases = (
            {"label": "Reserve balances elsewhere"},
            {"duplicate_row": True},
            {"cell_count": 4},
            {"last_attr": "header='not-an-id'"},
            {"last_attr": ""},
            {"last_attr": "header='t2r1c1'"},
            {"last_attr": "header='t2h1c3 t2h1c3'"},
            {"last_attr": "header='t2h1c3' headers='t2h1c3'"},
            {"header_text": "Monday Dec 24, 2025"},
            {"header_day": pd.Timestamp("2025-12-29", tz="UTC")},
            {"value": "N.A."},
            {"value": "-1"},
            {"value": "inf"},
        )
        for change in bad_cases:
            with self.subTest(change=change), self.assertRaises(VintageDataError):
                _parse_release_page(_page(day, **change), day)

    def test_full_loader_uses_only_injected_synthetic_pages_and_no_value_manifest(self):
        days = pd.date_range("2020-11-01", "2025-12-31", freq="W-THU", tz="UTC")
        self.assertEqual(len(days), EXPECTED_RELEASES)
        sources = {RELEASE_INDEX_URL: _synthetic_index(days)}
        for day in days:
            sources[_release_url(day)] = _page(
                day,
                template="old" if day.year == 2020 else "new",
                preceding_tables=2 if day.year == 2020 else 3,
                value="1,234.5",
            )
        requested: list[str] = []

        def fetch(url: str) -> bytes:
            requested.append(url)
            return sources[url]

        bundle = load_and_audit(fetcher=fetch, max_workers=4)
        self.assertEqual(len(bundle.observations), EXPECTED_RELEASES)
        self.assertEqual(len(requested), EXPECTED_RELEASES + 1)
        self.assertEqual(bundle.manifest["parser_version"], PARSER_VERSION)
        self.assertEqual(bundle.manifest["release_count"], EXPECTED_RELEASES)
        last_page = bundle.manifest["pages"][days[-1].date().isoformat()]
        self.assertEqual(last_page["sha256"], sha256(sources[_release_url(days[-1])]).hexdigest())
        self.assertEqual(last_page["parser_version"], PARSER_VERSION)
        self.assertNotIn("value", last_page)
        self.assertNotIn("observations", bundle.manifest)
        self.assertTrue(all("value" not in page for page in bundle.manifest["pages"].values()))
        json.dumps(bundle.manifest, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
