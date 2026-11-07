from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import unittest

from research.alt_common_flow_data import AltFunding, AltMinute, SYMBOLS
from research.alt_common_flow_sim import (
    _Regime, _Scenario, _funding_events, _simulate, FlowSimulationError,
)


UTC = timezone.utc


class FlowSimulationTests(unittest.TestCase):
    def test_fixed_signal_fill_funding_cost_and_exit_minute_exclusion(self) -> None:
        start = datetime(2021, 1, 1, 0, tzinfo=UTC)
        input_start = start - timedelta(hours=1)
        end = start + timedelta(days=1)
        bars = []
        for k in range(25 * 60):
            time = input_start + timedelta(minutes=k)
            burst = start <= time < start + timedelta(minutes=5)
            q = (60.0, 60.0, 20.0) if burst else (20.0, 20.0, 20.0)
            b = (0.0, 0.0, 10.0) if burst else (10.0, 10.0, 10.0)
            low = (90.0, 100.0, 100.0) if k == 66 else (100.0,) * 3
            if k == 96:  # This bar begins at exit and must not enter pressure DD.
                low = (1.0, 100.0, 100.0)
            bars.append(AltMinute(time, (100.0,) * 3, (100.0,) * 3,
                                  low, (100.0,) * 3, q, b))
        events = (
            AltFunding(start + timedelta(minutes=6, milliseconds=5), SYMBOLS[0], .01),
            AltFunding(start + timedelta(minutes=36, milliseconds=5), SYMBOLS[0], .10),
        )
        result = _simulate(bars, events, input_start=input_start, start=start,
                           end=end, require_funding_grid=False)
        formal = result["base_2x"]
        self.assertEqual(len(formal.trades), 1)
        trade = formal.trades[0]
        self.assertEqual(trade.entry_signal_time, start + timedelta(minutes=5))
        self.assertEqual(trade.entry_fill_time, start + timedelta(minutes=6))
        self.assertEqual(trade.exit_fill_time, start + timedelta(minutes=36))
        self.assertEqual(trade.direction, "long_all")
        self.assertLess(trade.funding_pnl, -60)
        self.assertGreater(trade.funding_pnl, -70)  # Exit-stamped 10% is excluded.
        self.assertLess(formal.max_drawdown, .10)  # Exit-minute low=1 is excluded.
        self.assertGreater(formal.max_drawdown, .06)  # Entry-minute low=90 counts.
        self.assertAlmostEqual(sum(trade.quantities) * 100.25, 20_000, places=5)
        self.assertEqual(len(formal.daily_records), 1)
        self.assertEqual(len(result["base_1x"].trades), 1)
        self.assertGreater(formal.max_drawdown, result["base_1x"].max_drawdown)
        self.assertLess(result["doubled_cost_2x"].final_equity, formal.final_equity)
        self.assertAlmostEqual(result["no_funding_credit_2x"].final_equity,
                               formal.final_equity, places=8)

    def test_funding_grid_group_accepts_distinct_millisecond_order(self) -> None:
        start = datetime(2021, 1, 1, tzinfo=UTC)
        end = start + timedelta(days=1)
        events = []
        for h in (0, 8, 16):
            anchor = start + timedelta(hours=h)
            events.extend(AltFunding(anchor + timedelta(milliseconds=ms), symbol, .001)
                          for ms, symbol in ((1, SYMBOLS[2]), (5, SYMBOLS[0]),
                                             (9, SYMBOLS[1])))
        self.assertEqual(len(_funding_events(events, start, end, require_grid=True)), 9)
        with self.assertRaises(FlowSimulationError):
            _funding_events(events[:-1] + events[:1], start, end, require_grid=True)

    def test_pending_open_and_sixty_minute_entry_cooldown(self) -> None:
        start = datetime(2021, 1, 1, tzinfo=UTC)
        first = start - timedelta(hours=1)
        end = start + timedelta(days=1)
        burst_starts = {start, start + timedelta(minutes=30),
                        start + timedelta(minutes=35), start + timedelta(hours=1)}
        bars = []
        for k in range(25 * 60):
            time = first + timedelta(minutes=k)
            burst = any(t <= time < t + timedelta(minutes=5) for t in burst_starts)
            q = (60.0, 60.0, 20.0) if burst else (20.0,) * 3
            b = (0.0, 0.0, 10.0) if burst else (10.0,) * 3
            bars.append(AltMinute(time, (100.0,) * 3, (100.0,) * 3,
                                  (100.0,) * 3, (100.0,) * 3, q, b))
        scenarios = _simulate(bars, (), input_start=first, start=start, end=end,
                              require_funding_grid=False)
        fills = [t.entry_fill_time for t in scenarios["base_2x"].trades]
        self.assertEqual(fills, [start + timedelta(minutes=6),
                                 start + timedelta(hours=1, minutes=6)])

    def test_mark_uses_three_simultaneous_extrema_and_same_minute_peak(self) -> None:
        time = datetime(2021, 1, 1, tzinfo=UTC)
        scenario = _Scenario("test", 2.0, .0025, False)
        scenario.start_day(time.date())
        scenario.enter(time, (100.0,) * 3, 1, time - timedelta(minutes=1), "unknown")
        bar = AltMinute(time, (100.0,) * 3, (110.0,) * 3,
                        (80.0,) * 3, (110.0,) * 3,
                        (1.0,) * 3, (.5,) * 3)
        scenario.mark(bar)
        self.assertGreater(scenario.last_regular, 11_000)
        self.assertGreater(scenario.max_drawdown, .40)
        self.assertTrue(math.isclose(scenario.day_max_drawdown,
                                     scenario.max_drawdown, rel_tol=1e-12))

    def test_regime_does_not_guess_without_history(self) -> None:
        state = _Regime()
        self.assertEqual(state.observe(datetime(2021, 1, 1, tzinfo=UTC),
                                       (1.0,) * 3, for_signal=True), "unknown")

    def test_fixed_warmup_supplies_all_sixty_prior_daily_vol_snapshots(self) -> None:
        state = _Regime()
        first = datetime(2020, 9, 1, tzinfo=UTC)
        stop = datetime(2021, 1, 1, tzinfo=UTC)
        count = int((stop - first).total_seconds() // 300)
        bucket = None
        for k in range(count + 1):
            time = first + timedelta(minutes=5 * k)
            price = 100 * math.exp(.000002 * k + .001 * math.sin(k / 37))
            bucket = state.observe(time, (price, price * 2, price * 3),
                                   for_signal=time == stop)
        self.assertIn(bucket, ("bull_high", "bull_low"))
        self.assertEqual(len([d for d, _ in state.daily_vol if d < stop.date()]), 60)


if __name__ == "__main__":
    unittest.main()
