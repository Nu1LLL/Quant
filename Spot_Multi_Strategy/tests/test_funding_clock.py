from datetime import datetime, timezone
import unittest

from research.crypto_blackbox.funding_clock import (
    FundingClockError, settled_hour, settled_hour_ms, unique_settlements,
)


class FundingClockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = int(datetime(2021, 1, 19, 8, tzinfo=timezone.utc).timestamp() * 1000)

    def test_subsecond_stamp_belongs_to_settlement_hour(self) -> None:
        self.assertEqual(settled_hour_ms(self.anchor + 5), self.anchor)
        self.assertEqual(settled_hour(self.anchor + 5),
                         datetime(2021, 1, 19, 8, tzinfo=timezone.utc))
        self.assertEqual(settled_hour_ms(self.anchor + 999), self.anchor)

    def test_invalid_timestamp_type_or_range_fails_closed(self) -> None:
        for stamp in (0, -1, 1.0, True, "1611043200005"):
            with self.subTest(stamp=stamp), self.assertRaises(FundingClockError):
                settled_hour_ms(stamp)  # type: ignore[arg-type]

    def test_later_than_one_second_and_duplicate_hour_fail_closed(self) -> None:
        with self.assertRaises(FundingClockError):
            settled_hour_ms(self.anchor + 1000)
        with self.assertRaises(FundingClockError):
            unique_settlements((self.anchor + 5, self.anchor + 12))
        self.assertEqual(unique_settlements((self.anchor + 5,
                                             self.anchor + 3_600_000 + 12)),
                         (self.anchor, self.anchor + 3_600_000))


if __name__ == "__main__":
    unittest.main()
