"""Synthetic tests for the bounded BTC archive feed; no real price read."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from research.btc_jump_reversal_data import (
    MANIFEST_PATH, _funding_url, _read_funding_month, iter_price_bars,
    read_pinned_manifest, validate_funding_sequence,
)
from research.btc_jump_reversal_sim import DataGateError, FundingRate, OOS_END_MS, OOS_START_MS
from research.btc_jump_reversal_runner import run_one_shot, retry_complete_transport
from research.crypto_blackbox.binance_vision_5m import ArchiveIntegrityError


def zipped(name: str, content: str) -> bytes:
    file = BytesIO()
    with ZipFile(file, "w", ZIP_DEFLATED) as archive:
        archive.writestr(name, content)
    return file.getvalue()


class ArchiveFeedTests(unittest.TestCase):
    def test_retry_transport_never_exposes_partial_attempt(self):
        attempts = 0

        def flaky(_url):
            nonlocal attempts
            attempts += 1
            yield b"first"
            if attempts == 1:
                raise ArchiveIntegrityError("transient connection failure")
            yield b"second"

        self.assertEqual(
            b"".join(retry_complete_transport(flaky)("https://example.test")),
            b"firstsecond",
        )
        self.assertEqual(attempts, 2)

    def test_pinned_manifest_is_complete_and_rejects_mutation(self):
        manifest = read_pinned_manifest()
        self.assertEqual(len(manifest), 72)
        self.assertEqual(next(iter(manifest)), "2020-01")
        self.assertEqual(next(reversed(manifest)), "2025-12")
        raw = MANIFEST_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(raw.replace("2021-01,um,", "2021-01,spot,", 1),
                            encoding="utf-8")
            with self.assertRaisesRegex(DataGateError, "invalid manifest pin"):
                read_pinned_manifest(path)

    def test_funding_checksum_schema_and_finite_rate(self):
        url, basename = _funding_url("2021-01")
        name = basename[:-4] + ".csv"
        content = (
            "calc_time,funding_interval_hours,last_funding_rate\n"
            "1609459200002,8,0.0002\n"
            "1609488000000,8,-0.0001\n"
        )
        data = zipped(name, content)
        checksum = f"{sha256(data).hexdigest()}  {basename}\n".encode()
        mapping = {url: data, url + ".CHECKSUM": checksum}
        fetch = lambda source: (mapping[source][i:i + 1024]
                                for i in range(0, len(mapping[source]), 1024))
        hashes = {}
        events = _read_funding_month("2021-01", fetch, audit_hashes=hashes)
        self.assertEqual([round(item.rate, 4) for item in events], [0.0002, -0.0001])
        self.assertEqual(hashes, {"2021-01": sha256(data).hexdigest()})
        mapping[url] = data + b"tampered"
        with self.assertRaisesRegex(DataGateError, "CHECKSUM"):
            _read_funding_month("2021-01", fetch)
        mapping[url] = zipped(name, content.replace("0.0002", "NaN"))
        mapping[url + ".CHECKSUM"] = (
            f"{sha256(mapping[url]).hexdigest()}  {basename}\n".encode()
        )
        with self.assertRaisesRegex(DataGateError, "nonfinite"):
            _read_funding_month("2021-01", fetch)

    def test_funding_sequence_requires_full_span_and_no_gaps(self):
        step = 8 * 60 * 60 * 1000
        events = [
            FundingRate(time_ms, 0.0001)
            for time_ms in range(OOS_START_MS, OOS_END_MS, step)
        ]
        self.assertEqual(validate_funding_sequence(events), events)
        with self.assertRaisesRegex(DataGateError, "does not span"):
            validate_funding_sequence(events[2:])
        with self.assertRaisesRegex(DataGateError, "gap"):
            validate_funding_sequence(events[:10] + events[12:])
        with self.assertRaisesRegex(DataGateError, "unsorted"):
            validate_funding_sequence(events[:10] + [events[9]] + events[10:])

    def test_price_iterator_refuses_unpinned_audit_before_revealing_price(self):
        # Existing monthly auditor has its own full-ZIP tests. Here we check
        # the additional frozen-manifest barrier before any price is emitted.
        from contextlib import contextmanager

        @contextmanager
        def fake_audit(*args, **kwargs):
            yield None, None, {"sha256": "0" * 64, "rows": 44640}

        with patch(
            "research.btc_jump_reversal_data._opened_verified_archive", fake_audit
        ):
            iterator = iter_price_bars(fetch_chunks=lambda url: ())
            with self.assertRaisesRegex(DataGateError, "pinned version"):
                next(iterator)

    def test_one_shot_closes_price_spool_on_simulation_error(self):
        class PriceIterator:
            closed = False

            def close(self):
                self.closed = True

        prices = PriceIterator()
        with (
            patch("research.btc_jump_reversal_runner.load_funding", return_value=[]),
            patch("research.btc_jump_reversal_runner.iter_price_bars", return_value=prices),
            patch("research.btc_jump_reversal_runner.simulate",
                  side_effect=DataGateError("synthetic failure")),
        ):
            with self.assertRaisesRegex(DataGateError, "synthetic failure"):
                run_one_shot(fetch_chunks=lambda url: ())
        self.assertTrue(prices.closed)


if __name__ == "__main__":
    unittest.main()
