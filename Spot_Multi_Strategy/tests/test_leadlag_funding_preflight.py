from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile
import unittest

from research.basis_hourly_preflight import PreflightError
from research.leadlag_funding_preflight import audit_one


class LeadLagFundingPreflightTests(unittest.TestCase):
    def test_hour_only_audit_does_not_parse_rate_and_rejects_off_hour(self):
        name = "BTCUSDT-fundingRate-2021-01.zip"
        start = datetime(2021, 1, 1, tzinfo=timezone.utc)

        def run(*, shifted: bool):
            lines = ["calc_time,funding_interval_hours,last_funding_rate"]
            for index in range(31 * 3):
                time = start + timedelta(hours=index * 8)
                stamp = int(time.timestamp() * 1000) + (60_000 if shifted and index == 4 else 0)
                lines.append(f"{stamp},8,NOT_A_RATE")
            buffer = BytesIO()
            with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
                archive.writestr(name[:-4] + ".csv", "\n".join(lines) + "\n")
            blob = buffer.getvalue()
            digest = sha256(blob).hexdigest()
            with patch("research.leadlag_funding_preflight._fetch",
                       side_effect=lambda url, **_: f"{digest}  {name}\n".encode()
                       if url.endswith(".CHECKSUM") else blob):
                return audit_one("BTCUSDT", "2021-01")

        self.assertEqual(run(shifted=False)["events"], 93)
        with self.assertRaisesRegex(PreflightError, "off-hour"):
            run(shifted=True)


if __name__ == "__main__":
    unittest.main()
