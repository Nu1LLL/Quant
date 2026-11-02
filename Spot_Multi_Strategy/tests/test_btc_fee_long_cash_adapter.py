"""Synthetic-only checks of the preregistered BTC-fee long/cash adapter."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.btc_fee_coinmetrics_data import BtcFeeDataError
from research.btc_fee_long_cash_adapter import (
    EXPECTED_WEEKS,
    FIRST_WEDNESDAY,
    LAST_WEDNESDAY,
    adapt_long_cash_observations,
)
from research.fed_reserves_crypto_sim import _direction, _validate_observations


@dataclass(frozen=True)
class FeeObservation:
    release_date: pd.Timestamp
    observed_date: pd.Timestamp
    value: Decimal


def _fixture() -> list[FeeObservation]:
    dates = pd.date_range(FIRST_WEDNESDAY, LAST_WEDNESDAY, freq="W-WED", tz="UTC")
    assert len(dates) == EXPECTED_WEEKS
    raw = [Decimal("100") for _ in dates]
    # Jan 4 uses Dec 30 (i=9) versus Dec 2 (i=5): long.
    raw[5], raw[9] = Decimal("5"), Decimal("10")
    # Jan 11 uses Jan 6 (i=10) versus Dec 9 (i=6): cash.
    raw[6], raw[10] = Decimal("20"), Decimal("15")
    # Jan 18 uses Jan 13 (i=11) versus Dec 16 (i=7): tie, cash.
    raw[7], raw[11] = Decimal("30"), Decimal("30")
    return [FeeObservation(day + pd.Timedelta(days=1), day, value)
            for day, value in zip(dates, raw, strict=True)]


class BtcFeeLongCashAdapterTests(unittest.TestCase):
    def test_exact_registered_grid_and_binary_old_ledger_directions(self) -> None:
        rows = _fixture()
        adapted = adapt_long_cash_observations(rows)
        self.assertEqual(len(adapted), EXPECTED_WEEKS)
        self.assertEqual(adapted[0].observed_date, FIRST_WEDNESDAY)
        self.assertEqual(adapted[-1].observed_date, LAST_WEDNESDAY)
        self.assertTrue(all(item.value == 0 for item in adapted[:4]))
        self.assertEqual(adapted[9].value - adapted[5].value, 1)
        self.assertEqual(adapted[10].value - adapted[6].value, 0)
        self.assertEqual(adapted[11].value - adapted[7].value, 0)
        for index in range(4, EXPECTED_WEEKS):
            self.assertEqual(
                adapted[index].value - adapted[index - 4].value,
                int(rows[index].value > rows[index - 4].value),
            )

        releases, values = _validate_observations(adapted)
        for monday, expected in (("2021-01-04", 1), ("2021-01-11", 0),
                                 ("2021-01-18", 0)):
            with self.subTest(monday=monday):
                self.assertEqual(_direction(releases, values,
                                            pd.Timestamp(monday, tz="UTC")), expected)

        for monday in pd.date_range("2021-01-04", "2025-12-29", freq="W-MON", tz="UTC"):
            latest = bisect_right(releases, monday - pd.Timedelta(days=4)) - 1
            expected = int(rows[latest].value > rows[latest - 4].value)
            self.assertEqual(_direction(releases, values, monday), expected)

    def test_first_monday_does_not_see_future_wednesday(self) -> None:
        adapted = adapt_long_cash_observations(_fixture())
        releases, values = _validate_observations(adapted)
        monday = pd.Timestamp("2021-01-04", tz="UTC")
        self.assertEqual(releases[9], pd.Timestamp("2020-12-31", tz="UTC"))
        self.assertEqual(releases[10], pd.Timestamp("2021-01-07", tz="UTC"))
        self.assertEqual(monday - pd.Timedelta(days=4), releases[9])
        self.assertEqual(_direction(releases, values, monday), 1)
        # The Jan 6 observation would signal cash, but is unavailable on Jan 4.
        self.assertEqual(values[10] - values[6], 0)

    def test_raw_decimal_comparison_retains_tiny_strict_increase(self) -> None:
        rows = _fixture()
        rows[5] = FeeObservation(rows[5].release_date, rows[5].observed_date,
                                 Decimal("1.00000000000000000001"))
        rows[9] = FeeObservation(rows[9].release_date, rows[9].observed_date,
                                 Decimal("1.00000000000000000002"))
        adapted = adapt_long_cash_observations(rows)
        self.assertEqual(adapted[9].value - adapted[5].value, 1)

    def test_rejects_missing_week_wrong_release_or_bad_amount(self) -> None:
        rows = _fixture()
        with self.assertRaises(BtcFeeDataError):
            adapt_long_cash_observations(rows[:-1])

        bad_day = list(rows)
        bad_day[20] = FeeObservation(rows[20].release_date,
                                     rows[20].observed_date + pd.Timedelta(days=7),
                                     rows[20].value)
        with self.assertRaises(BtcFeeDataError):
            adapt_long_cash_observations(bad_day)

        late = list(rows)
        late[20] = FeeObservation(rows[20].release_date + pd.Timedelta(days=1),
                                  rows[20].observed_date, rows[20].value)
        with self.assertRaises(BtcFeeDataError):
            adapt_long_cash_observations(late)

        for bad_value in (Decimal("NaN"), Decimal("Infinity"), Decimal("-0.1"),
                          1.0, True):
            with self.subTest(bad_value=bad_value):
                bad = list(rows)
                bad[20] = FeeObservation(rows[20].release_date,
                                         rows[20].observed_date, bad_value)
                with self.assertRaises(BtcFeeDataError):
                    adapt_long_cash_observations(bad)

    def test_rejects_naive_or_nonmidnight_timestamps(self) -> None:
        rows = _fixture()
        for field, timestamp in (("observed_date", pd.Timestamp("2021-03-17")),
                                 ("release_date", pd.Timestamp("2021-03-18 00:00:01", tz="UTC"))):
            with self.subTest(field=field):
                bad = list(rows)
                item = rows[20]
                bad[20] = FeeObservation(
                    timestamp if field == "release_date" else item.release_date,
                    timestamp if field == "observed_date" else item.observed_date,
                    item.value,
                )
                with self.assertRaises(BtcFeeDataError):
                    adapt_long_cash_observations(bad)


if __name__ == "__main__":
    unittest.main()
