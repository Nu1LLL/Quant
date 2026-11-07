from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile
import unittest

from research.basis_hourly_preflight import PreflightError
from research.leadlag_5m_preflight import audit_one


class LeadLagPreflightTests(unittest.TestCase):
    def test_checksummed_full_grid_and_gap_rejection(self):
        name = "BTCUSDT-5m-2020-12.zip"
        start = int(datetime(2020, 12, 1, tzinfo=timezone.utc).timestamp() * 1000)

        def payload(*, gap: bool) -> bytes:
            lines = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"]
            for i in range(31 * 288):
                stamp = start + i * 300_000 + (300_000 if gap and i == 23 else 0)
                lines.append(f"{stamp},1,2,0.5,1,10,{stamp+299999},10,1,5,5,0")
            body = BytesIO()
            with ZipFile(body, "w", ZIP_DEFLATED) as archive:
                archive.writestr(name[:-4] + ".csv", "\n".join(lines) + "\n")
            return body.getvalue()

        for gap in (False, True):
            blob = payload(gap=gap)
            digest = sha256(blob).hexdigest()
            with patch("research.leadlag_5m_preflight._fetch",
                       side_effect=lambda url, **_: f"{digest}  {name}\n".encode()
                       if url.endswith(".CHECKSUM") else blob):
                if gap:
                    with self.assertRaisesRegex(PreflightError, "timestamp gap"):
                        audit_one("BTCUSDT", "2020-12")
                else:
                    row = audit_one("BTCUSDT", "2020-12")
                    self.assertEqual(row["bars"], 31 * 288)
                    self.assertEqual(row["sha256"], digest)


if __name__ == "__main__":
    unittest.main()
