"""Synthetic-only tests for the frozen fear/greed signal adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.fng_crypto_archive import FNGIntegrityError
from research.fng_crypto_adapter import (
    ANCHOR_VALUE,
    EXPECTED_DAYS,
    EXPECTED_WEEKS,
    FIRST_ANCHOR,
    FIRST_DAY,
    LAST_DAY,
    adapt_fng_observations,
    signal_fingerprint,
)
from research.fed_reserves_crypto_sim import _direction, _validate_observations


@dataclass(frozen=True)
class SampleObservation:
    day: pd.Timestamp
    value: int


def _fixture() -> list[SampleObservation]:
    days = pd.date_range(FIRST_DAY, LAST_DAY, freq="D", tz="UTC")
    assert len(days) == EXPECTED_DAYS
    scores = (25, 75, 26, 74, 0, 100, 50)
    return [SampleObservation(day, scores[((day - FIRST_DAY).days // 7) % len(scores)]
                              if day.weekday() == 2 else 50)
            for day in days]


class FngCryptoAdapterTests(unittest.TestCase):
    def test_all_261_frozen_ledger_directions_and_fingerprint(self) -> None:
        rows = _fixture()
        adapted = adapt_fng_observations(rows)
        fingerprint = signal_fingerprint(rows)
        self.assertEqual(len(adapted), 4 + EXPECTED_WEEKS)
        self.assertEqual(len(fingerprint), EXPECTED_WEEKS)
        for index, proxy in enumerate(adapted[:4]):
            self.assertEqual(proxy.observed_date, FIRST_ANCHOR + index * pd.Timedelta(weeks=1))
            self.assertEqual(proxy.release_date, proxy.observed_date + pd.Timedelta(days=1))
            self.assertEqual(proxy.value, ANCHOR_VALUE)

        releases, values = _validate_observations(adapted)
        for index, entry in enumerate(fingerprint):
            with self.subTest(index=index):
                wednesday = FIRST_DAY + index * pd.Timedelta(weeks=1)
                monday = wednesday + pd.Timedelta(days=5)
                raw = rows[(wednesday - FIRST_DAY).days].value
                expected = 1 if raw <= 25 else -1 if raw >= 75 else 0
                self.assertEqual(entry, {
                    "day": wednesday.strftime("%Y-%m-%d"),
                    "value": raw,
                    "direction": expected,
                })
                self.assertEqual(adapted[index + 4].observed_date, wednesday)
                self.assertEqual(adapted[index + 4].release_date,
                                 wednesday + pd.Timedelta(days=1))
                self.assertEqual(values[index + 4] - values[index], expected)
                self.assertEqual(_direction(releases, values, monday), expected)

        self.assertEqual(FIRST_DAY + pd.Timedelta(days=5),
                         pd.Timestamp("2021-01-04", tz="UTC"))
        self.assertEqual(LAST_DAY + pd.Timedelta(days=5),
                         pd.Timestamp("2025-12-29", tz="UTC"))

    def test_exact_boundaries_and_four_day_lag(self) -> None:
        rows = _fixture()
        fingerprint = signal_fingerprint(rows)
        self.assertEqual([entry["direction"] for entry in fingerprint[:7]],
                         [1, -1, 0, 0, 1, -1, 0])
        releases, values = _validate_observations(adapt_fng_observations(rows))
        first_monday = pd.Timestamp("2021-01-04", tz="UTC")
        self.assertEqual(releases[4], first_monday - pd.Timedelta(days=4))
        self.assertGreater(releases[5], first_monday)
        self.assertEqual(_direction(releases, values, first_monday), 1)
        self.assertEqual(_direction(releases, values,
                                    pd.Timestamp("2021-01-11", tz="UTC")), -1)
        last_monday = pd.Timestamp("2025-12-29", tz="UTC")
        self.assertEqual(releases[-1], last_monday - pd.Timedelta(days=4))
        self.assertEqual(_direction(releases, values, last_monday),
                         fingerprint[-1]["direction"])

    def test_non_wednesday_values_have_no_signal_effect(self) -> None:
        rows = _fixture()
        changed = list(rows)
        for index, item in enumerate(changed):
            if item.day.weekday() != 2:
                changed[index] = SampleObservation(item.day, 0)
        self.assertEqual(signal_fingerprint(changed), signal_fingerprint(rows))
        self.assertEqual(adapt_fng_observations(changed), adapt_fng_observations(rows))

    def test_rejects_missing_duplicate_reordered_and_invalid_rows(self) -> None:
        rows = _fixture()
        for bad in (rows[:-1], rows[1:], "not observations"):
            with self.subTest(kind="count", length=len(bad)):
                with self.assertRaises(FNGIntegrityError):
                    adapt_fng_observations(bad)

        for changed_day in (rows[10].day + pd.Timedelta(days=1),
                            rows[10].day.tz_localize(None),
                            rows[10].day + pd.Timedelta(seconds=1),
                            rows[10].day.tz_convert("Europe/Berlin")):
            with self.subTest(day=changed_day):
                bad = list(rows)
                bad[10] = SampleObservation(changed_day, rows[10].value)
                with self.assertRaises(FNGIntegrityError):
                    signal_fingerprint(bad)

        for bad_value in (-1, 101, 25.0, True, "25", None):
            with self.subTest(value=bad_value):
                bad = list(rows)
                bad[10] = SampleObservation(rows[10].day, bad_value)
                with self.assertRaises(FNGIntegrityError):
                    adapt_fng_observations(bad)


if __name__ == "__main__":
    unittest.main()
