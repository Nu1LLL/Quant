"""Synthetic-only contract tests; never request actual FNG history."""

import json
from hashlib import sha256
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.fng_crypto_archive import (
    END,
    EXPECTED_DAYS,
    FNGIntegrityError,
    FNG_URL,
    START,
    _Payload,
    load_and_audit,
)


def _fixture(*, mutate=None, metadata=None):
    dates = pd.date_range(START, END, freq="D", tz="UTC")
    records = [
        {"timestamp": str(int(day.timestamp()) + 3600), "value": str(i % 101)}
        for i, day in enumerate(dates)
    ]
    if mutate is not None:
        mutate(records)
    return json.dumps({
        "name": "Synthetic Index",
        "data": records,
        "metadata": {"error": None} if metadata is None else metadata,
    }).encode("utf-8")


class FNGCryptoArchiveTests(unittest.TestCase):
    def test_one_fixed_get_and_daily_values(self):
        body = _fixture()
        calls = []

        def fetcher(url):
            calls.append(url)
            return _Payload(body, 200, "application/json", url)

        bundle = load_and_audit(fetcher=fetcher)
        self.assertEqual(calls, [FNG_URL])
        self.assertEqual(len(bundle.observations), EXPECTED_DAYS)
        self.assertEqual(bundle.observations[0].day, START)
        self.assertEqual(bundle.observations[-1].day, END)
        self.assertEqual(bundle.observations[0].value, 0)
        self.assertEqual(bundle.observations[100].value, 100)
        self.assertEqual(bundle.manifest["response_bytes"], len(body))
        self.assertEqual(bundle.manifest["sha256"], sha256(body).hexdigest())
        self.assertEqual(bundle.manifest["http_status"], 200)
        self.assertEqual(bundle.manifest["coverage_days"], 1821)
        self.assertIn("+00:00", bundle.manifest["fetched_at_utc"])
        json.dumps(bundle.manifest, allow_nan=False)

    def test_records_outside_frozen_range_do_not_change_grid(self):
        def mutate(records):
            records.insert(0, {"timestamp": "1608595200", "value": "1"})
            records.append({"timestamp": "1766620800", "value": "99"})

        bundle = load_and_audit(fetcher=lambda _url: _fixture(mutate=mutate))
        self.assertEqual(len(bundle.observations), EXPECTED_DAYS)
        self.assertEqual(bundle.observations[0].day, START)
        self.assertEqual(bundle.observations[-1].day, END)

    def test_missing_or_duplicate_registered_day_fails(self):
        with self.assertRaisesRegex(FNGIntegrityError, "incomplete FNG UTC daily grid"):
            load_and_audit(fetcher=lambda _url: _fixture(mutate=lambda rows: rows.pop(500)))

        def duplicate(rows):
            rows.append(dict(rows[500]))

        with self.assertRaisesRegex(FNGIntegrityError, "duplicate FNG UTC day"):
            load_and_audit(fetcher=lambda _url: _fixture(mutate=duplicate))

    def test_value_is_strict_integer_string_with_registered_bounds(self):
        for value in (-1, 101, 22.5, True, None, "2.5", "NaN", "-1", 25):
            with self.subTest(value=value), self.assertRaises(FNGIntegrityError):
                def mutate(rows):
                    rows[42]["value"] = value

                load_and_audit(fetcher=lambda _url: _fixture(mutate=mutate))
        for value in ("0", "100"):
            with self.subTest(value=value):
                def mutate(rows):
                    rows[42]["value"] = value

                bundle = load_and_audit(fetcher=lambda _url: _fixture(mutate=mutate))
                self.assertEqual(bundle.observations[42].value, int(value))

    def test_timestamp_must_be_unix_seconds_string(self):
        for timestamp in (1609286400000, "1609286400000", 1609286400, "banana", None):
            with self.subTest(timestamp=timestamp), self.assertRaises(FNGIntegrityError):
                def mutate(rows):
                    rows[0]["timestamp"] = timestamp

                load_and_audit(fetcher=lambda _url: _fixture(mutate=mutate))

    def test_metadata_error_and_json_structure_fail(self):
        with self.assertRaisesRegex(FNGIntegrityError, "metadata.error"):
            load_and_audit(fetcher=lambda _url: _fixture(metadata={"error": "denied"}))
        with self.assertRaisesRegex(FNGIntegrityError, "metadata.error"):
            load_and_audit(fetcher=lambda _url: _fixture(metadata={}))
        for body in (b"not json", b"[]", b'{"data":[],"metadata":{"error":null}}'):
            with self.subTest(body=body), self.assertRaises(FNGIntegrityError):
                load_and_audit(fetcher=lambda _url, body=body: body)
        with self.assertRaisesRegex(FNGIntegrityError, "duplicate JSON key"):
            load_and_audit(fetcher=lambda _url: b'{"data":[],"data":[],"metadata":{"error":null}}')
        with self.assertRaisesRegex(FNGIntegrityError, "nonstandard JSON number"):
            load_and_audit(fetcher=lambda _url: b'{"data":[],"metadata":{"error":NaN}}')

    def test_http_and_source_integrity_fail_with_provenance(self):
        body = _fixture()
        for payload in (
            _Payload(body, 503, "application/json", FNG_URL),
            _Payload(body, 200, "text/html", FNG_URL),
            _Payload(body, 200, "application/json", "https://other.example/fng"),
        ):
            with self.subTest(payload=payload.status_code), self.assertRaises(FNGIntegrityError) as caught:
                load_and_audit(fetcher=lambda _url, payload=payload: payload)
            self.assertEqual(caught.exception.manifest["sha256"], sha256(body).hexdigest())
            self.assertEqual(caught.exception.manifest["response_bytes"], len(body))

    def test_parse_failure_retains_nonpayload_provenance(self):
        body = _fixture(mutate=lambda rows: rows.pop(0))
        with self.assertRaises(FNGIntegrityError) as caught:
            load_and_audit(fetcher=lambda _url: body)
        self.assertIsNotNone(caught.exception.manifest)
        self.assertEqual(caught.exception.manifest["sha256"], sha256(body).hexdigest())
        self.assertNotIn("data", caught.exception.manifest)


if __name__ == "__main__":
    unittest.main()
