"""Synthetic retry and immutable checksum checks; no market data downloads."""

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

from research.frozen_source_transport import (
    FetchReceipt, IntegrityFailure, TransportFailure,
    audit_checksum_manifest, fetch_pinned,
)


URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2023-05.zip.CHECKSUM"


class FrozenTransportTests(unittest.TestCase):
    def test_transient_tls_failure_retries_same_endpoint_and_hashes_result(self):
        failed = SimpleNamespace(returncode=35, stdout=b"", stderr=b"TLS error")
        succeeded = SimpleNamespace(returncode=0, stdout=b"verified bytes", stderr=b"")
        digest = sha256(succeeded.stdout).hexdigest()
        with patch("research.frozen_source_transport.subprocess.run",
                   side_effect=(failed, succeeded)) as run, patch(
                       "research.frozen_source_transport.time.sleep"):
            receipt = fetch_pinned(URL, expected_sha256=digest)
        self.assertEqual(receipt.attempts, 2)
        self.assertEqual(receipt.sha256, digest)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][-1], URL)
        self.assertEqual(run.call_args_list[1].args[0][-1], URL)

    def test_content_mismatch_is_terminal_and_bad_hosts_rejected(self):
        result = SimpleNamespace(returncode=0, stdout=b"revised", stderr=b"")
        with patch("research.frozen_source_transport.subprocess.run", return_value=result) as run:
            with self.assertRaises(IntegrityFailure):
                fetch_pinned(URL, expected_sha256="0" * 64)
            self.assertEqual(run.call_count, 1)
        oversized = SimpleNamespace(returncode=63, stdout=b"", stderr=b"too large")
        with patch("research.frozen_source_transport.subprocess.run", return_value=oversized) as run:
            with self.assertRaises(IntegrityFailure):
                fetch_pinned(URL, max_bytes=1)
            self.assertEqual(run.call_count, 1)
        with self.assertRaises(ValueError):
            fetch_pinned("http://data.binance.vision/archive.zip")
        with self.assertRaises(ValueError):
            fetch_pinned("https://data.binance.vision.evil.example/archive.zip")
        failed = SimpleNamespace(returncode=35, stdout=b"", stderr=b"TLS error")
        with patch("research.frozen_source_transport.subprocess.run", return_value=failed), patch(
                "research.frozen_source_transport.time.sleep"):
            with self.assertRaises(TransportFailure):
                fetch_pinned(URL, attempts=2)

    def test_manifest_audit_reads_only_checksum_sidecars(self):
        name = "BTCUSDT-1d-2023-05.zip"
        price_url = URL.removesuffix(".CHECKSUM")
        pin = "a" * 64
        body = f"month,url,official_zip_sha256\n2023-05,{price_url},{pin}\n".encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_bytes(body)
            checksum = f"{pin}  {name}\n".encode()
            with patch("research.frozen_source_transport.fetch_pinned",
                       return_value=FetchReceipt(checksum, sha256(checksum).hexdigest(), 2)) as fetch:
                audit = audit_checksum_manifest(path, sha256(body).hexdigest())
            self.assertEqual((audit.checked, audit.retries, audit.all_verified), (1, 1, True))
            self.assertEqual(fetch.call_args.args[0], URL)
            with self.assertRaises(IntegrityFailure):
                audit_checksum_manifest(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
