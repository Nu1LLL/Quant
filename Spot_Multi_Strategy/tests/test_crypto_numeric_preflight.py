"""Source invariant checks stay separate from strategy calculations."""

import unittest

from research.crypto_numeric_preflight import _row_flags


class NumericPreflightTests(unittest.TestCase):
    def setUp(self):
        self.bar = ["0", "100", "102", "99", "101", "500", "299999", "50000",
                    "10", "250", "25000", "0"]

    def test_valid_bar(self):
        self.assertEqual(_row_flags(self.bar), ())

    def test_detects_official_archive_failure_pattern(self):
        self.bar[9] = "6635.3"
        self.assertEqual(_row_flags(self.bar), ("taker_base_exceeds_total",))

    def test_reports_all_independent_invariants(self):
        self.bar[2] = "98"
        self.bar[9] = "501"
        self.bar[10] = "50001"
        self.assertEqual(_row_flags(self.bar), (
            "ohlc_bounds", "taker_base_exceeds_total", "taker_quote_exceeds_total"))

    def test_nonfinite_and_bad_shape(self):
        self.bar[4] = "nan"
        self.assertEqual(_row_flags(self.bar), ("nonfinite",))
        self.assertEqual(_row_flags(self.bar[:-1]), ("row_shape",))


if __name__ == "__main__":
    unittest.main()
