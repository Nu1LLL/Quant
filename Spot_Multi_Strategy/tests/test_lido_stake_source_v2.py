"""Synthetic timestamp semantics for the new Lido source version."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from research.eth_defi_tvl_source import SourceFailure
from research import lido_stake_source_v2 as source


def fixture(bad_clock=False):
    first = datetime(2020, 12, 20, tzinfo=timezone.utc)
    items = []
    for i in range(263):
        hour = 23 if i < 17 else 22 if i < 24 else 0
        if bad_clock and i == 0:
            hour = 21
        stamp = int((first + timedelta(days=i * 7, hours=hour)).timestamp())
        items.append([stamp, 1000.0 + i])
    body = json.dumps({"name": "Lido", "chainTvls": {"Ethereum": {"tokens": [
        {"date": stamp, "tokens": {"WETH": amount}} for stamp, amount in items
    ]}}}).encode()
    content_sha = sha256(json.dumps(items, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()
    timestamp_sha = sha256(json.dumps([x[0] for x in items],
                                      separators=(",", ":")).encode()).hexdigest()
    return body, content_sha, timestamp_sha


class LidoSourceV2Test(unittest.TestCase):
    def test_late_sunday_timestamp_preserves_sunday_date(self):
        body, content, timestamps = fixture()
        with patch.object(source, "STAKED_SHA", content), patch.object(source, "TIMESTAMP_SHA", timestamps):
            series, audit = source.parse_stake(body)
        self.assertEqual(len(series), 263)
        self.assertEqual(str(next(iter(series))), "2020-12-20")
        self.assertEqual(audit["utc_clock_counts"], {"00:00:00": 239,
                                                      "22:00:00": 7,
                                                      "23:00:00": 17})

    def test_unregistered_clock_fails_even_with_matching_hashes(self):
        body, content, timestamps = fixture(bad_clock=True)
        with patch.object(source, "STAKED_SHA", content), patch.object(source, "TIMESTAMP_SHA", timestamps):
            with self.assertRaises(SourceFailure):
                source.parse_stake(body)


if __name__ == "__main__":
    unittest.main()
