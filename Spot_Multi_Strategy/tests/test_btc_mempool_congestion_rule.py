"""Causality and outage behavior of the preregistered weekly rule."""

from datetime import date, timedelta
import unittest

from research.btc_mempool_congestion_sim import weekly_target


class MempoolRuleTest(unittest.TestCase):
    def test_equal_threshold_buys_and_future_day_is_ignored(self):
        monday = date(2021, 1, 4)
        friday = monday - timedelta(days=3)
        dates = [friday - timedelta(days=i) for i in range(34, -1, -1)]
        values = {day: (1_000_000 if i < 28 else 2_000_000)
                  for i, day in enumerate(dates)}
        counts = {day: 96 for day in dates}
        values[friday + timedelta(days=1)] = 10_000_000_000
        counts[friday + timedelta(days=1)] = 96
        self.assertEqual(weekly_target(monday, values, counts), (1, False))
        values[friday + timedelta(days=1)] = 0
        self.assertEqual(weekly_target(monday, values, counts), (1, False))

    def test_gap_forces_cash_instead_of_omitting_week(self):
        monday = date(2021, 1, 4)
        friday = monday - timedelta(days=3)
        dates = [friday - timedelta(days=i) for i in range(35)]
        values = {day: 10_000_000 for day in dates}
        counts = {day: 96 for day in dates}
        counts[dates[17]] = 89
        self.assertEqual(weekly_target(monday, values, counts), (0, True))


if __name__ == "__main__":
    unittest.main()
