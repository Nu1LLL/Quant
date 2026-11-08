"""Synthetic as-of and no-backfill checks for the forward CPI source."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo
import json
import sqlite3
import unittest

from research.crypto_blackbox.cleveland_nowcast_prospective import (
    capture_bytes, parse_current, read_asof,
)


ET = ZoneInfo("America/New_York")
VISUAL = {"anchorradius": "4", "anchorbgcolor": "000000",
          "anchorbordercolor": "000000", "anchorborderthickness": "1"}


def _stamp(day: int) -> int:
    return int(datetime(2026, 11, day, 11, 0, tzinfo=ET).timestamp() * 1000)


def _chart(name: str, labels: list[str], forecast: list[str],
           actual: list[str]):
    blank = lambda: [{"value": "", "tooltext": ""} for _ in labels]
    datasets = [{"seriesname": "CPI Inflation",
                 "data": [{"value": v, "tooltext": "forecast"} for v in forecast]}]
    datasets.extend({"seriesname": "unused", "data": blank()} for _ in range(3))
    datasets.append({"seriesname": "Actual CPI Inflation",
                     "data": [{"value": v, "tooltext": "actual", **VISUAL}
                              if v else {"value": "", "tooltext": ""} for v in actual]})
    datasets.extend({"seriesname": "unused", "data": blank()} for _ in range(3))
    return {"chart": {"subcaption": name,
                      "yaxisname": "Month-over-month percent change"},
            "categories": [{"category": [{"label": v} for v in labels]}],
            "dataset": datasets}


def _body(actual_released: bool) -> bytes:
    filler = [{"chart": {"subcaption": f"{2013 + i // 12}-{1 + i % 12}"}}
              for i in range(98)]
    october = _chart("2026-10", ["11/11", "11/12"], ["0.2", ""],
                     ["", "0.1" if actual_released else ""])
    november = _chart("2026-11", ["11/11", "11/12"], ["0.3", "0.4"], ["", ""])
    return json.dumps([*filler, october, november], separators=(",", ":")).encode()


class ProspectiveNowcastTests(unittest.TestCase):
    def test_forecast_before_release_then_link_actual(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nowcast.sqlite"
            before = capture_bytes(path, _body(False), _stamp(11))
            self.assertEqual((before.status, before.forecasts, before.actuals),
                             ("fresh", 2, 0))
            self.assertEqual(read_asof(path, _stamp(11) - 1),
                             {"forecasts": [], "actuals": []})
            after = capture_bytes(path, _body(True), _stamp(12))
            self.assertEqual((after.status, after.actuals, after.linked_actuals),
                             ("fresh", 1, 1))
            snapshot = read_asof(path, _stamp(12))
            self.assertEqual(len(snapshot["actuals"]), 1)
            self.assertIsNotNone(snapshot["actuals"][0]["prior_forecast_id"])
            self.assertEqual(len(snapshot["forecasts"]), 3)
            self.assertEqual(capture_bytes(path, _body(True), _stamp(12)).status,
                             "already_captured_today")
            with sqlite3.connect(path) as db:
                with self.assertRaises(sqlite3.DatabaseError):
                    db.execute("DELETE FROM forecasts")

    def test_late_initial_capture_does_not_backfill_forecast(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nowcast.sqlite"
            result = capture_bytes(path, _body(True), _stamp(12))
            self.assertEqual((result.actuals, result.linked_actuals), (1, 0))
            snapshot = read_asof(path, _stamp(12))
            self.assertEqual([x["month"] for x in snapshot["forecasts"]],
                             ["2026-11"])
            self.assertIsNone(snapshot["actuals"][0]["prior_forecast_id"])

    def test_invalid_actual_renderer_and_duplicate_json_fail_closed(self):
        raw = _body(True).replace(b'"anchorborderthickness":"1"',
                                  b'"unexpected":"1"', 1)
        with self.assertRaises(ValueError):
            parse_current(raw, _stamp(12))
        duplicated = _body(False).replace(
            b'"yaxisname":"Month-over-month percent change"',
            b'"yaxisname":"Month-over-month percent change","yaxisname":"wrong"', 1)
        with self.assertRaises(ValueError):
            parse_current(duplicated, _stamp(11))


if __name__ == "__main__":
    unittest.main()
