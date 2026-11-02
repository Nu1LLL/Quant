import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))

from research.funding_data import validate_funding_rates


class FundingDataTests(unittest.TestCase):
    def test_validation_sorts_and_converts(self):
        frame = pd.DataFrame(
            {
                "funding_time": [
                    "2026-01-01T08:00:00Z", "2026-01-01T00:00:00Z"
                ],
                "funding_rate": ["0.0001", "-0.0002"],
            }
        )
        validated = validate_funding_rates(frame)
        self.assertTrue(validated["funding_time"].is_monotonic_increasing)
        self.assertEqual(validated["funding_rate"].dtype.kind, "f")

    def test_duplicate_timestamp_is_rejected(self):
        frame = pd.DataFrame(
            {
                "funding_time": ["2026-01-01T00:00:00Z"] * 2,
                "funding_rate": [0.0, 0.1],
            }
        )
        with self.assertRaises(ValueError):
            validate_funding_rates(frame)


if __name__ == "__main__":
    unittest.main()
