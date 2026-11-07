"""Synthetic timing and sign checks; no historical returns are read."""

from datetime import datetime, timedelta, timezone
import unittest

from research.alt_common_flow_data import AltMinute
from research.alt_common_flow_signal import CommonFlowSignal


START = datetime(2021, 1, 1, tzinfo=timezone.utc)


def feed_window(state, number, totals, buy_fraction):
    result = None
    for minute in range(5):
        q = tuple(x / 5 for x in totals)
        b = tuple(x * y for x, y in zip(q, buy_fraction))
        bar = AltMinute(
            START + timedelta(minutes=number * 5 + minute),
            (1.0,) * 3, (1.0,) * 3, (1.0,) * 3, (1.0,) * 3,
            q, b,
        )
        result = state.consume(bar)
        if minute < 4:
            assert result is None
    return result


class FlowSignalTests(unittest.TestCase):
    def test_current_burst_excluded_from_baseline_and_shorts_common_buying(self):
        state = CommonFlowSignal()
        for n in range(12):
            self.assertIsNone(feed_window(state, n, (100, 100, 100), (.5, .5, .5)))
        signal = feed_window(state, 12, (200, 200, 100), (.8, .8, .5))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, -1)
        self.assertEqual(signal.time, START + timedelta(minutes=65))
        self.assertEqual(signal.volume_multiple[:2], (2.0, 2.0))
        self.assertEqual(signal.qualifying_coins, 2)
        self.assertEqual(state.finish(), 13)

    def test_symmetric_common_selling_buys_basket(self):
        state = CommonFlowSignal()
        for n in range(12):
            feed_window(state, n, (100, 100, 100), (.5, .5, .5))
        signal = feed_window(state, 12, (200, 200, 100), (.1, .1, .5))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, 1)
        self.assertLess(signal.imbalance[0], -.2)

    def test_one_coin_burst_and_incomplete_history_do_not_signal(self):
        state = CommonFlowSignal()
        for n in range(11):
            feed_window(state, n, (100, 100, 100), (.5, .5, .5))
        self.assertIsNone(feed_window(state, 11, (200, 200, 100), (.8, .8, .5)))
        state = CommonFlowSignal()
        for n in range(12):
            feed_window(state, n, (100, 100, 100), (.5, .5, .5))
        self.assertIsNone(feed_window(state, 12, (200, 100, 100), (.8, .5, .5)))

    def test_missing_minute_fails_closed(self):
        state = CommonFlowSignal()
        feed_window(state, 0, (100, 100, 100), (.5, .5, .5))
        t = START + timedelta(minutes=6)
        bar = AltMinute(t, (1,) * 3, (1,) * 3, (1,) * 3, (1,) * 3,
                        (1,) * 3, (0.5,) * 3)
        with self.assertRaisesRegex(ValueError, "missing"):
            state.consume(bar)


if __name__ == "__main__":
    unittest.main()
