"""Synthetic-only tests; these never request Coin Metrics or market history."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.btc_fee_coinmetrics_data import (
    BtcFeeDataError,
    EXPECTED_DAYS,
    EXPECTED_WEDNESDAYS,
    FIRST_DAY,
    LAST_DAY,
    START_URL,
    _Payload,
    load_and_audit,
)


def _encode(page: object) -> bytes:
    return json.dumps(page, separators=(",", ":"), allow_nan=True).encode()


def _rows() -> list[dict]:
    grid = pd.date_range(FIRST_DAY, LAST_DAY, freq="D", tz="UTC")
    assert len(grid) == EXPECTED_DAYS
    return [
        {"asset": "btc", "time": day.strftime("%Y-%m-%dT00:00:00.000000000Z"),
         "FeeTotNtv": "1.25"}
        for day in grid
    ]


def _fetcher(rows: list[dict], *, split: int | None = None,
             status_code: int = 200, next_url: str | None = None):
    if split is None:
        pages = {START_URL: _encode({"data": rows})}
    else:
        next_url = next_url or f"{START_URL}&next_page_token=synthetic-page-2"
        pages = {
            START_URL: _encode({"data": rows[:split], "next_page_url": next_url}),
            next_url: _encode({"data": rows[split:]}),
        }

    def fetch(url: str) -> _Payload:
        return _Payload(pages[url], status_code if url == START_URL else 200, 1)

    return fetch, pages


class BtcFeeCoinMetricsDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _rows()

    def test_full_grid_raw_rolling_sums_dates_lag_and_safe_manifest(self):
        fetch, pages = _fetcher(self.rows)
        bundle = load_and_audit(fetcher=fetch)
        self.assertEqual(len(bundle.observations), EXPECTED_WEDNESDAYS)
        self.assertEqual(bundle.observations[0].observed_date,
                         pd.Timestamp("2020-10-28", tz="UTC"))
        self.assertEqual(bundle.observations[-1].observed_date, LAST_DAY)
        self.assertEqual(bundle.observations[0].release_date,
                         pd.Timestamp("2020-10-29", tz="UTC"))
        self.assertEqual(bundle.observations[-1].release_date,
                         pd.Timestamp("2026-01-01", tz="UTC"))
        self.assertEqual(bundle.observations[0].value, Decimal("35.00"))
        self.assertTrue(all(item.observed_date.weekday() == 2 and
                            item.release_date - item.observed_date == pd.Timedelta(days=1)
                            for item in bundle.observations))
        self.assertEqual(bundle.manifest["daily_row_count"], EXPECTED_DAYS)
        self.assertEqual(bundle.manifest["weekly_observation_count"], EXPECTED_WEDNESDAYS)
        self.assertFalse(bundle.manifest["independent_oos"])
        self.assertEqual(bundle.manifest["pages"][0]["sha256"],
                         sha256(pages[START_URL]).hexdigest())
        self.assertEqual(bundle.manifest["pages"][0]["bytes"], len(pages[START_URL]))
        self.assertEqual(bundle.manifest["pages"][0]["row_count"], EXPECTED_DAYS)
        self.assertNotIn("1.25", json.dumps(bundle.manifest))
        self.assertNotIn("35.00", json.dumps(bundle.manifest))

    def test_28_day_window_is_inclusive_and_drops_oldest_day(self):
        rows = deepcopy(self.rows)
        rows[0]["FeeTotNtv"] = "7"
        rows[27]["FeeTotNtv"] = "3"
        first, second = load_and_audit(fetcher=_fetcher(rows)[0]).observations[:2]
        self.assertEqual(first.value, Decimal("42.50"))  # 26*1.25 + 7 + 3
        self.assertEqual(second.value, Decimal("36.75"))  # Oct 1 left; Oct 28 remains.

    def test_tiny_strict_increase_survives_loader_exact_sum(self):
        rows = deepcopy(self.rows)
        for row in rows:
            row["FeeTotNtv"] = "1"
        # 2020-11-25 is Wednesday i=4; its added 1e-28 belongs to R_4
        # but not to R_0. The default Decimal context would erase it.
        rows[55]["FeeTotNtv"] = "1.0000000000000000000000000001"
        observations = load_and_audit(fetcher=_fetcher(rows)[0]).observations
        self.assertEqual(observations[0].value, Decimal("28"))
        self.assertEqual(observations[4].value,
                         Decimal("28.0000000000000000000000000001"))
        self.assertGreater(observations[4].value, observations[0].value)

    def test_extreme_decimal_scale_or_literal_is_rejected(self):
        for bad in ("1e-65", "0e+309", "1." + "0" * 128):
            with self.subTest(bad=bad):
                rows = deepcopy(self.rows)
                rows[0]["FeeTotNtv"] = bad
                with self.assertRaises(BtcFeeDataError):
                    load_and_audit(fetcher=_fetcher(rows)[0])

    def test_first_monday_uses_december_30_and_december_2_not_future(self):
        bundle = load_and_audit(fetcher=_fetcher(self.rows)[0])
        cutoff = pd.Timestamp("2021-01-04", tz="UTC") - pd.Timedelta(days=4)
        visible = [item for item in bundle.observations if item.release_date <= cutoff]
        self.assertEqual(visible[-1].observed_date, pd.Timestamp("2020-12-30", tz="UTC"))
        self.assertEqual(visible[-5].observed_date, pd.Timestamp("2020-12-02", tz="UTC"))
        self.assertEqual(visible[-1].release_date, cutoff)
        later = next(item for item in bundle.observations
                     if item.observed_date == pd.Timestamp("2021-01-06", tz="UTC"))
        self.assertGreater(later.release_date, cutoff)

    def test_official_next_page_url_is_followed_and_each_page_fingerprinted(self):
        fetch, pages = _fetcher(self.rows, split=1000)
        calls = []

        def tracked(url: str):
            calls.append(url)
            return fetch(url)

        bundle = load_and_audit(fetcher=tracked)
        self.assertEqual(calls, list(pages))
        self.assertEqual([page["row_count"] for page in bundle.manifest["pages"]],
                         [1000, 918])
        self.assertEqual(bundle.manifest["pages"][1]["first_day"], "2023-06-28")

    def test_missing_duplicate_wrong_order_and_out_of_range_days_fail(self):
        variants = {}
        variants["missing"] = self.rows[:10] + self.rows[11:]
        variants["duplicate"] = self.rows[:10] + [self.rows[10]] + self.rows[10:]
        variants["reordered"] = self.rows[:10] + [self.rows[11], self.rows[10]] + self.rows[12:]
        wrong = deepcopy(self.rows)
        wrong[0]["time"] = "2020-09-30T00:00:00Z"
        variants["out of range"] = wrong
        for label, rows in variants.items():
            with self.subTest(label=label), self.assertRaises(BtcFeeDataError):
                load_and_audit(fetcher=_fetcher(rows)[0])

    def test_wrong_asset_metric_time_and_bad_fee_fail(self):
        mutations = [
            ("asset", "eth"),
            ("FeeTotNtv", "-1"),
            ("FeeTotNtv", "NaN"),
            ("FeeTotNtv", "Infinity"),
            ("FeeTotNtv", "1e999999"),
            ("FeeTotNtv", None),
            ("FeeTotNtv", 1.25),
            ("time", "2020-10-01T01:00:00Z"),
            ("time", "2020-10-01T00:00:00+00:00"),
            ("time", "2020-10-01"),
        ]
        for field, bad in mutations:
            with self.subTest(field=field, bad=bad):
                rows = deepcopy(self.rows)
                rows[0][field] = bad
                with self.assertRaises(BtcFeeDataError):
                    load_and_audit(fetcher=_fetcher(rows)[0])
        rows = deepcopy(self.rows)
        del rows[0]["FeeTotNtv"]
        with self.assertRaises(BtcFeeDataError):
            load_and_audit(fetcher=_fetcher(rows)[0])

    def test_http_json_schema_and_duplicate_json_key_fail(self):
        with self.assertRaises(BtcFeeDataError):
            load_and_audit(fetcher=_fetcher(self.rows, status_code=429)[0])
        for body in (b"{", b'[]', b'{"data":{}}',
                     b'{"data":[],"data":[]}', b'{"data":[]}'):
            with self.subTest(body=body), self.assertRaises(BtcFeeDataError):
                load_and_audit(fetcher=lambda _url, body=body: body)

    def test_pagination_source_switch_repeated_url_and_cross_page_duplicate_fail(self):
        malicious = [
            "http://community-api.coinmetrics.io/v4/timeseries/asset-metrics?"
            + START_URL.split("?", 1)[1] + "&next_page_token=x",
            "https://evil.example/v4/timeseries/asset-metrics?"
            + START_URL.split("?", 1)[1] + "&next_page_token=x",
            f"{START_URL}&metrics=Other&next_page_token=x",
            f"{START_URL}&next_page_token=",
        ]
        for url in malicious:
            with self.subTest(url=url), self.assertRaises(BtcFeeDataError):
                load_and_audit(fetcher=_fetcher(self.rows, split=1000, next_url=url)[0])

        loop_page = _encode({"data": self.rows[:1000],
                             "next_page_url": f"{START_URL}&next_page_token=loop"})
        loop_next = _encode({"data": self.rows[1000:],
                             "next_page_url": f"{START_URL}&next_page_token=loop"})
        def loop_fetch(url: str):
            return loop_page if url == START_URL else loop_next
        with self.assertRaises(BtcFeeDataError):
            load_and_audit(fetcher=loop_fetch)

        duplicate_next = self.rows[999:]
        next_url = f"{START_URL}&next_page_token=duplicate"
        payloads = {START_URL: _encode({"data": self.rows[:1000],
                                         "next_page_url": next_url}),
                    next_url: _encode({"data": duplicate_next})}
        with self.assertRaises(BtcFeeDataError):
            load_and_audit(fetcher=lambda url: payloads[url])

    def test_pagination_token_and_url_must_agree_and_cannot_be_orphaned(self):
        next_url = f"{START_URL}&next_page_token=second"
        pages = {
            START_URL: _encode({"data": self.rows[:1000],
                                "next_page_url": next_url,
                                "next_page_token": "different"}),
            next_url: _encode({"data": self.rows[1000:]}),
        }
        with self.assertRaises(BtcFeeDataError):
            load_and_audit(fetcher=lambda url: pages[url])
        orphan = _encode({"data": self.rows, "next_page_token": "orphan"})
        with self.assertRaises(BtcFeeDataError):
            load_and_audit(fetcher=lambda _url: orphan)
        no_token_url = f"{START_URL}"
        missing_token = _encode({"data": self.rows[:1000],
                                 "next_page_url": no_token_url})
        with self.assertRaises(BtcFeeDataError):
            load_and_audit(fetcher=lambda _url: missing_token)


if __name__ == "__main__":
    unittest.main()
