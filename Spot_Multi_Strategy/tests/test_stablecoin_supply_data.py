"""Synthetic-only tests for the preregistered stablecoin supply reader."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.stablecoin_supply_data import (
    CATALOG_URL,
    EXPECTED_DAYS,
    EXPECTED_WEDNESDAYS,
    FIRST_DAY,
    HISTORY_URLS,
    LAST_DAY,
    StablecoinDataError,
    _Payload,
    load_and_audit,
)


def _encode(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=True).encode()


def _fixture() -> tuple[dict, dict[str, list[dict]]]:
    catalog = {"peggedAssets": [
        {"id": "1", "name": "Tether", "symbol": "USDT", "pegType": "peggedUSD"},
        {"id": "2", "name": "USD Coin", "symbol": "USDC", "pegType": "peggedUSD"},
    ]}
    grid = pd.date_range(FIRST_DAY, LAST_DAY, freq="D", tz="UTC")
    assert len(grid) == EXPECTED_DAYS
    histories = {}
    for symbol, base in (("USDT", 1000), ("USDC", 2000)):
        histories[symbol] = [
            {
                "date": str(int(day.timestamp())),
                "totalCirculating": {"peggedUSD": base + index},
                # Deliberately different: the reader must not substitute USD
                # market value for the registered raw circulation units.
                "totalCirculatingUSD": {"peggedUSD": 100000000000 + index},
            }
            for index, day in enumerate(grid)
        ]
    return catalog, histories


def _fetcher(catalog: dict, histories: dict[str, list[dict]], *,
             status_url: str | None = None, status_code: int = 200):
    payloads = {CATALOG_URL: _encode(catalog)}
    payloads.update({url: _encode(histories[symbol]) for symbol, url in HISTORY_URLS.items()})

    def fetch(url: str) -> _Payload:
        return _Payload(payloads[url], status_code if url == status_url else 200, 1)

    return fetch, payloads


class StablecoinSupplyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog, cls.histories = _fixture()

    def test_full_grid_wednesdays_proxy_release_and_raw_units(self):
        fetch, payloads = _fetcher(self.catalog, self.histories)
        bundle = load_and_audit(fetcher=fetch)
        self.assertEqual(len(bundle.observations), EXPECTED_WEDNESDAYS)
        self.assertEqual(bundle.observations[0].observed_date, FIRST_DAY)
        self.assertEqual(bundle.observations[-1].observed_date, LAST_DAY)
        self.assertEqual(bundle.observations[0].release_date,
                         pd.Timestamp("2020-11-04", tz="UTC"))
        self.assertEqual(bundle.observations[-1].release_date,
                         pd.Timestamp("2026-01-07", tz="UTC"))
        self.assertTrue(all(item.observed_date.weekday() == 2 and
                            item.release_date - item.observed_date == pd.Timedelta(days=7)
                            for item in bundle.observations))
        self.assertEqual(bundle.observations[0].value, 3000.0)
        self.assertEqual(bundle.observations[1].value, 3014.0)
        self.assertEqual(bundle.manifest["histories"]["USDT"]["sha256"],
                         sha256(payloads[HISTORY_URLS["USDT"]]).hexdigest())
        self.assertEqual(bundle.manifest["histories"]["USDT"]["used_row_count"],
                         EXPECTED_DAYS)
        self.assertFalse(bundle.manifest["independent_oos"])
        self.assertNotIn("100000000000", json.dumps(bundle.manifest))

    def test_first_monday_cannot_use_later_wednesday(self):
        bundle = load_and_audit(fetcher=_fetcher(self.catalog, self.histories)[0])
        cutoff = pd.Timestamp("2021-01-04", tz="UTC") - pd.Timedelta(days=4)
        visible = [item for item in bundle.observations if item.release_date <= cutoff]
        self.assertEqual(visible[-1].observed_date, pd.Timestamp("2020-12-23", tz="UTC"))
        self.assertEqual(visible[-5].observed_date, pd.Timestamp("2020-11-25", tz="UTC"))
        later = next(item for item in bundle.observations
                     if item.observed_date == pd.Timestamp("2020-12-30", tz="UTC"))
        self.assertGreater(later.release_date, cutoff)

    def test_catalog_id_symbol_or_peg_mismatch_is_rejected(self):
        for field, bad in (("symbol", "USDC"), ("pegType", "peggedEUR"),
                           ("id", "3")):
            with self.subTest(field=field):
                catalog = deepcopy(self.catalog)
                catalog["peggedAssets"][0][field] = bad
                with self.assertRaises(StablecoinDataError):
                    load_and_audit(fetcher=_fetcher(catalog, self.histories)[0])

    def test_catalog_duplicate_id_is_rejected(self):
        catalog = deepcopy(self.catalog)
        catalog["peggedAssets"].append(dict(catalog["peggedAssets"][0]))
        with self.assertRaises(StablecoinDataError):
            load_and_audit(fetcher=_fetcher(catalog, self.histories)[0])

    def test_missing_or_duplicate_day_is_rejected(self):
        for mutation in ("missing", "duplicate"):
            with self.subTest(mutation=mutation):
                histories = {symbol: list(rows) for symbol, rows in self.histories.items()}
                if mutation == "missing":
                    del histories["USDC"][10]
                else:
                    histories["USDC"].insert(10, deepcopy(histories["USDC"][10]))
                with self.assertRaises(StablecoinDataError):
                    load_and_audit(fetcher=_fetcher(self.catalog, histories)[0])

    def test_non_midnight_or_non_unix_date_is_rejected(self):
        for date in ("1603843201", "2020-10-28", 1603843200):
            with self.subTest(date=date):
                histories = {symbol: list(rows) for symbol, rows in self.histories.items()}
                histories["USDT"][0] = {**histories["USDT"][0], "date": date}
                with self.assertRaises(StablecoinDataError):
                    load_and_audit(fetcher=_fetcher(self.catalog, histories)[0])

    def test_missing_wrong_unit_or_invalid_amount_is_rejected(self):
        for circulation in ({"peggedEUR": 1000}, {"peggedUSD": 0},
                            {"peggedUSD": -1}, {"peggedUSD": True},
                            {"peggedUSD": "1000"}, {"peggedUSD": math.nan}):
            with self.subTest(circulation=circulation):
                histories = {symbol: list(rows) for symbol, rows in self.histories.items()}
                histories["USDT"][0] = {
                    **histories["USDT"][0], "totalCirculating": circulation,
                }
                with self.assertRaises(StablecoinDataError):
                    load_and_audit(fetcher=_fetcher(self.catalog, histories)[0])

    def test_http_and_json_failures_are_data_errors(self):
        fetch, _ = _fetcher(self.catalog, self.histories,
                           status_url=HISTORY_URLS["USDC"], status_code=503)
        with self.assertRaises(StablecoinDataError):
            load_and_audit(fetcher=fetch)

        def invalid_json(url: str) -> bytes:
            if url == CATALOG_URL:
                return b"not JSON"
            return b"[]"

        with self.assertRaises(StablecoinDataError):
            load_and_audit(fetcher=invalid_json)


if __name__ == "__main__":
    unittest.main()
