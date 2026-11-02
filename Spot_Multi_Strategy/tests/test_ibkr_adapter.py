import sys
import unittest
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))

from research.ibkr_adapter import audit_bars


class IbkrAdapterTests(unittest.TestCase):
    def test_audit_accepts_ordered_valid_bars(self):
        bars = [
            {"date": "100", "open": 10, "high": 12, "low": 9, "close": 11},
            {"date": "160", "open": 11, "high": 13, "low": 10, "close": 12},
        ]
        self.assertTrue(audit_bars(bars)["passed"])

    def test_audit_rejects_invalid_ohlc(self):
        bars = [{"date": "100", "open": 10, "high": 9, "low": 8, "close": 11}]
        self.assertFalse(audit_bars(bars)["passed"])


if __name__ == "__main__":
    unittest.main()
