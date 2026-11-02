"""Synthetic-only H.4.1 PIT vintage tests: no Federal Reserve values are fetched."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.fed_reserves_vintage_data import (
    EXPECTED_RELEASES,
    RELEASE_INDEX_URL,
    ROW_LABEL,
    UNIT,
    VintageDataError,
    _Payload,
    _parse_release_index,
    _parse_release_page,
    _release_url,
    _http_fetch,
    load_and_audit,
)


def _dates() -> pd.DatetimeIndex:
    # This synthetic weekly calendar has exactly 269 Thursdays, nine in
    # Nov/Dec 2020 and 52 in each full year; it is not official release data.
    return pd.date_range("2020-11-01", "2025-12-31", freq="W-THU", tz="UTC")


def _index(dates=None) -> bytes:
    dates = _dates() if dates is None else dates
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


def _page(
    release, *, template="new", label=ROW_LABEL, value="1,234.5",
    cell_count=5, header_day=None, header_text=None, unit=UNIT,
    title_day=None, duplicate=False, header_attr=True,
):
    release = pd.Timestamp(release)
    observed = header_day or release - pd.Timedelta(days=1)
    title_day = title_day or release
    header_text = header_text or f"Wednesday {observed:%B %d, %Y}"
    old = template == "old"
    identifier = "t3c2" if old else "t2h1c3"
    attr = "headers" if old else "header"
    label_tag = "th" if old else "td"
    label_id = " id='t3r16'" if old else " id='t2r16c1'"
    last_attr = f" {attr}='{identifier}'" if header_attr else ""
    cells = [f"<{label_tag}{label_id}>{label}</{label_tag}>"]
    cells.extend("<td>0</td>" for _ in range(cell_count - 2))
    cells.append(f"<td{last_attr}>{value}</td>")
    row = "<tr>" + "".join(cells) + "</tr>"
    if duplicate:
        row += row
    source = (
        "<html><head><title>H.4.1 Federal Reserve</title></head><body>"
        f"<div>For release at 4:30 p.m. Eastern Time {title_day:%B %d, %Y}</div>"
        "<h2>Table 1. Factors Affecting Reserve Balances of Depository Institutions</h2>"
        f"<p>{unit}</p><table><tr><th id='{identifier}'>{header_text}</th></tr>"
        + row + "</table></body></html>"
    )
    return source.encode()


class H41VintageTests(unittest.TestCase):
    def test_official_index_shape_and_fixed_269_window(self):
        days = _parse_release_index(_index())
        self.assertEqual(len(days), EXPECTED_RELEASES)
        self.assertEqual(days[0], pd.Timestamp("2020-11-05", tz="UTC"))
        self.assertEqual(days[-1], pd.Timestamp("2025-12-25", tz="UTC"))

    def test_index_rejects_missing_duplicate_wrong_month_and_count(self):
        payload = json.loads(_index())
        mutants = []
        missing = json.loads(json.dumps(payload))
        missing[0]["Months"][0]["Dates"].pop()
        mutants.append(missing)
        duplicate = json.loads(json.dumps(payload))
        duplicate[0]["Months"][0]["Dates"][-1] = duplicate[0]["Months"][0]["Dates"][0]
        mutants.append(duplicate)
        wrong_month = json.loads(json.dumps(payload))
        wrong_month[0]["Months"][0]["Dates"][0] = "20251101"
        mutants.append(wrong_month)
        absent_month = json.loads(json.dumps(payload))
        absent_month[0]["Months"].pop()
        mutants.append(absent_month)
        for mutant in mutants:
            with self.subTest(mutant=mutant[0]["Months"][0]["MonthValue"]):
                with self.assertRaises(VintageDataError):
                    _parse_release_index(json.dumps(mutant).encode())

    def test_old_and_new_html_headers_take_only_last_value(self):
        day = pd.Timestamp("2025-12-25", tz="UTC")
        for template in ("old", "new"):
            with self.subTest(template=template):
                observation, metadata = _parse_release_page(
                    _page(day, template=template, value="2,345.75"), day
                )
                self.assertEqual(observation.release_date, day)
                self.assertEqual(
                    observation.observed_date, pd.Timestamp("2025-12-24", tz="UTC")
                )
                self.assertEqual(observation.value, 2345.75)
                self.assertEqual(metadata["row_cell_count"], 5)
                self.assertEqual(metadata["unit"], UNIT)

    def test_structure_and_date_mismatches_fail_closed(self):
        day = pd.Timestamp("2025-12-25", tz="UTC")
        bad_cases = (
            {"label": "Reserve balances elsewhere"},
            {"cell_count": 4},
            {"unit": "Billions of dollars"},
            {"duplicate": True},
            {"header_attr": False},
            {"header_text": "Monday December 24, 2025"},
            {"header_day": pd.Timestamp("2025-12-25", tz="UTC")},
            {"title_day": pd.Timestamp("2025-12-18", tz="UTC")},
            {"value": "N.A."},
            {"value": "-1"},
            {"value": "inf"},
        )
        for change in bad_cases:
            with self.subTest(change=change):
                with self.assertRaises(VintageDataError):
                    _parse_release_page(_page(day, **change), day)

    def test_full_loader_pins_dates_and_hashes_without_persisting_values(self):
        days = _dates()
        sources = {RELEASE_INDEX_URL: _index()}
        for n, day in enumerate(days):
            sources[_release_url(day)] = _page(day, value=f"{n + 1:,}")
        requested = []

        def fetch(url):
            requested.append(url)
            return sources[url]

        result = load_and_audit(fetcher=fetch, max_workers=4)
        self.assertEqual(len(result.observations), EXPECTED_RELEASES)
        self.assertEqual(result.observations[0].value, 1)
        self.assertEqual(result.observations[-1].value, EXPECTED_RELEASES)
        self.assertEqual(len(requested), EXPECTED_RELEASES + 1)
        manifest = result.manifest
        self.assertEqual(manifest["release_count"], EXPECTED_RELEASES)
        self.assertEqual(manifest["release_index"]["sha256"], sha256(sources[RELEASE_INDEX_URL]).hexdigest())
        last_page = manifest["pages"][days[-1].date().isoformat()]
        self.assertEqual(last_page["observed_date"], "2025-12-24")
        self.assertEqual(last_page["http_status"], 200)
        self.assertEqual(last_page["http_attempts"], 1)
        self.assertEqual(last_page["sha256"], sha256(sources[_release_url(days[-1])]).hexdigest())
        self.assertEqual(last_page["label"], ROW_LABEL)
        self.assertNotIn("value", last_page)
        self.assertNotIn("observations", manifest)
        self.assertTrue(all("value" not in page for page in manifest["pages"].values()))
        json.dumps(manifest, allow_nan=False)

    def test_http_failure_and_duplicate_observations_are_rejected(self):
        days = _dates()
        sources = {RELEASE_INDEX_URL: _index()}
        for day in days:
            sources[_release_url(day)] = _page(day)
        bad_url = _release_url(days[-1])
        sources[bad_url] = _Payload(b"not found", 404)
        with self.assertRaisesRegex(VintageDataError, "HTTP 404"):
            load_and_audit(fetcher=sources.__getitem__, max_workers=2)
        sources[bad_url] = _page(days[-1], header_day=days[-2] - pd.Timedelta(days=1))
        with self.assertRaisesRegex(VintageDataError, "duplicate Wednesday"):
            load_and_audit(fetcher=sources.__getitem__, max_workers=2)

    def test_fixed_transport_retries_only_transient_failures(self):
        class Response:
            def __init__(self, status):
                self.status_code = status
                self.content = b"fixture"

        scripted = [requests.exceptions.Timeout(), Response(503), Response(200)]

        def get(_url, *, timeout):
            self.assertEqual(timeout, (10, 45))
            event = scripted.pop(0)
            if isinstance(event, Exception):
                raise event
            return event

        with patch("research.fed_reserves_vintage_data.requests.get", side_effect=get) as called, \
             patch("research.fed_reserves_vintage_data.time.sleep") as pause:
            payload = _http_fetch("https://www.federalreserve.gov/fixture")
        self.assertEqual(payload.body, b"fixture")
        self.assertEqual(payload.attempts, 3)
        self.assertEqual(called.call_count, 3)
        self.assertEqual([call.args[0] for call in pause.call_args_list], [0.5, 1.0])

        with patch("research.fed_reserves_vintage_data.requests.get", return_value=Response(404)) as called, \
             patch("research.fed_reserves_vintage_data.time.sleep") as pause:
            payload = _http_fetch("https://www.federalreserve.gov/fixture")
        self.assertEqual(payload.status_code, 404)
        self.assertEqual(called.call_count, 1)
        pause.assert_not_called()

        with patch("research.fed_reserves_vintage_data.requests.get", return_value=Response(429)) as called, \
             patch("research.fed_reserves_vintage_data.time.sleep") as pause:
            payload = _http_fetch("https://www.federalreserve.gov/fixture")
        self.assertEqual(payload.status_code, 429)
        self.assertEqual(payload.attempts, 4)
        self.assertEqual(called.call_count, 4)
        self.assertEqual(pause.call_count, 3)


if __name__ == "__main__":
    unittest.main()
