"""Synthetic-only checks for the frozen BTC/ETH pairs accounting engine."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import math
from statistics import stdev
import unittest

from research.btc_eth_pairs_sim import (
    BASE_ADVERSE_RATE, FEE_RATE, PairSimulationDataError,
    SignalQualityFailure, _HourlyState, _Scenario, _Spec, _simulate,
)


UTC = timezone.utc
START = datetime(2024, 1, 2, tzinfo=UTC)
INPUT_START = START - timedelta(hours=5)
END = START + timedelta(days=1)
TEST_SPEC = _Spec(INPUT_START, START, END, 4, 5, False)


@dataclass(frozen=True)
class Bar:
    time: datetime
    btc_open: float
    btc_high: float
    btc_low: float
    btc_close: float
    eth_open: float
    eth_high: float
    eth_low: float
    eth_close: float


@dataclass(frozen=True)
class Funding:
    time: datetime
    symbol: str
    rate: float


def flat_bar(time: datetime, btc: float = 100.0, eth: float = 100.0,
             *, btc_low: float | None = None,
             eth_high: float | None = None) -> Bar:
    return Bar(time, btc, btc, btc if btc_low is None else btc_low, btc,
               eth, eth if eth_high is None else eth_high, eth, eth)


def synthetic_bars():
    """One small OOS day; a single large synthetic BTC return crosses +2."""
    closes = [(100.0, 100.0)]
    hours = int((END - INPUT_START) / timedelta(hours=1))
    for hour in range(1, hours + 1):
        btc_return = 0.05 if hour == 6 else 0.001 * ((hour % 4) - 1.5)
        eth_return = 0.001 * ((hour * 2 % 5) - 2)
        last_btc, last_eth = closes[-1]
        closes.append((last_btc * math.exp(btc_return),
                       last_eth * math.exp(eth_return)))
    for minute in range(hours * 60):
        hour, minute_of_hour = divmod(minute, 60)
        time = INPUT_START + timedelta(minutes=minute)
        open_btc, open_eth = closes[hour]
        close_btc, close_eth = closes[hour + 1] if minute_of_hour == 59 else closes[hour]
        yield Bar(time,
                  open_btc, max(open_btc, close_btc), min(open_btc, close_btc), close_btc,
                  open_eth, max(open_eth, close_eth), min(open_eth, close_eth), close_eth)


def midnight_bars():
    """A formed pair enters before midnight, then stays open across funding."""
    end = START + timedelta(days=2)
    hours = int((end - INPUT_START) / timedelta(hours=1))
    closes = [(100.0, 100.0)]
    formation_btc = {2: 0.001, 3: -0.001, 4: 0.002, 5: -0.002}
    formation_eth = {2: 0.0005, 3: -0.0002, 4: -0.0005, 5: 0.0002}
    for hour in range(1, hours + 1):
        rb = formation_btc.get(hour, 0.0)
        re = formation_eth.get(hour, 0.0)
        if hour == 26:  # Jan 2 21:00 crossing; entry Jan 2 22:01.
            rb = 0.05
        if hour == 29:  # Completed Jan 2 23:59 minute; used by Jan 3 funding.
            rb = 0.01
        last_btc, last_eth = closes[-1]
        closes.append((last_btc * math.exp(rb), last_eth * math.exp(re)))
    for minute in range(hours * 60):
        hour, minute_of_hour = divmod(minute, 60)
        time = INPUT_START + timedelta(minutes=minute)
        open_btc, open_eth = closes[hour]
        close_btc, close_eth = closes[hour + 1] if minute_of_hour == 59 else closes[hour]
        yield Bar(time,
                  open_btc, max(open_btc, close_btc), min(open_btc, close_btc), close_btc,
                  open_eth, max(open_eth, close_eth), min(open_eth, close_eth), close_eth)


class PairIndexTests(unittest.TestCase):
    def test_score_uses_additive_standardized_returns_and_fixed_formation(self):
        state = _HourlyState(TEST_SPEC)
        btc = [100.0, 101.0, 100.5, 102.0, 101.0, 106.0]
        eth = [100.0, 99.6, 100.2, 99.7, 100.4, 100.5]
        for index in range(1, 6):
            time = INPUT_START + timedelta(hours=index)
            result = state.observe(time, btc[index - 1], eth[index - 1])
            self.assertIsNone(result)
        time = START + timedelta(hours=1)
        before, observed, regime = state.observe(time, btc[5], eth[5])
        rb = [math.log(btc[i] / btc[i - 1]) for i in range(1, 5)]
        re = [math.log(eth[i] / eth[i - 1]) for i in range(1, 5)]
        mu_b, mu_e = sum(rb) / 4, sum(re) / 4
        sd_b, sd_e = stdev(rb), stdev(re)
        distances = []
        ib = ie = 1.0
        for x, y in zip(rb, re):
            ib += (x - mu_b) / sd_b
            ie += (y - mu_e) / sd_e
            distances.append(ib - ie)
        mu_d, sd_d = sum(distances) / 4, stdev(distances)
        expected_before = (distances[-1] - mu_d) / sd_d
        ib += (math.log(btc[5] / btc[4]) - mu_b) / sd_b
        ie += (math.log(eth[5] / eth[4]) - mu_e) / sd_e
        self.assertAlmostEqual(before, expected_before)
        self.assertAlmostEqual(observed, ((ib - ie) - mu_d) / sd_d)
        self.assertEqual(regime, "unknown")

    def test_zero_leg_sigma_rejects_whole_cycle(self):
        state = _HourlyState(TEST_SPEC)
        for index in range(1, 6):
            time = INPUT_START + timedelta(hours=index)
            if index == 5:
                with self.assertRaises(SignalQualityFailure):
                    state.observe(time, 100.0, 100.0)
            else:
                state.observe(time, 100.0, 100.0)


class AccountingTests(unittest.TestCase):
    def test_two_leg_sizing_fees_funding_and_conservative_minute_mark(self):
        scenario = _Scenario("synthetic", 2.0, BASE_ADVERSE_RATE, False)
        scenario.start_day(START.date())
        scenario.enter(START, 100.0, 100.0, 1, START - timedelta(hours=1), "bull_low")
        pos = scenario.position
        assert pos is not None
        self.assertAlmostEqual(pos.quantity_btc * pos.entry_btc, 10_000.0)
        self.assertAlmostEqual(pos.quantity_eth * pos.entry_eth, 10_000.0)
        self.assertAlmostEqual(scenario.cash, 10_000.0 - 2 * FEE_RATE * 10_000.0)
        scenario.fund("BTCUSDT", 0.001, 100.0, 100.0)
        scenario.fund("ETHUSDT", 0.002, 100.0, 100.0)
        scenario.mark((100.0, 100.0, 90.0, 100.0,
                       100.0, 110.0, 100.0, 100.0))
        regular = scenario.last_regular
        adverse = (scenario.cash + pos.quantity_btc * (90.0 - pos.entry_btc)
                   - pos.quantity_eth * (110.0 - pos.entry_eth))
        self.assertAlmostEqual(scenario.day_min_adverse_factor, adverse / 10_000.0)
        self.assertAlmostEqual(scenario.day_max_drawdown,
                               1 - adverse / max(10_000.0, regular))
        scenario.exit(START + timedelta(hours=1), 100.0, 100.0)
        trade = scenario.trades[0]
        self.assertAlmostEqual(trade.net_pnl, scenario.cash - 10_000.0)
        self.assertAlmostEqual(trade.funding_pnl,
                               -0.001 * pos.quantity_btc * 100.0
                               + 0.002 * pos.quantity_eth * 100.0)
        self.assertAlmostEqual(trade.fee_pnl,
                               -FEE_RATE * (20_000.0
                                            + pos.quantity_btc * 100.0 * (1 - BASE_ADVERSE_RATE)
                                            + pos.quantity_eth * 100.0 * (1 + BASE_ADVERSE_RATE)))

    def test_current_minute_regular_close_sets_peak_before_adverse_mark(self):
        scenario = _Scenario("synthetic", 2.0, BASE_ADVERSE_RATE, False)
        scenario.start_day(START.date())
        scenario.enter(START, 100.0, 100.0, 1, START - timedelta(hours=1), "bull_low")
        pos = scenario.position
        assert pos is not None
        values = (100.0, 110.0, 90.0, 110.0,
                  100.0, 110.0, 90.0, 90.0)
        scenario.mark(values)
        regular = scenario.last_regular
        adverse = (scenario.cash + pos.quantity_btc * (90.0 - pos.entry_btc)
                   - pos.quantity_eth * (110.0 - pos.entry_eth))
        self.assertGreater(regular, 10_000.0)
        self.assertAlmostEqual(scenario.day_max_drawdown, 1 - adverse / regular)
        self.assertAlmostEqual(scenario.max_drawdown, scenario.day_max_drawdown)

    def test_full_synthetic_stream_delayed_fill_forced_exit_and_scenarios(self):
        funding_time = START + timedelta(hours=3, milliseconds=2)
        funding = [Funding(funding_time, "BTCUSDT", 0.001),
                   Funding(funding_time, "ETHUSDT", 0.002)]
        results = _simulate(synthetic_bars(), funding, spec=TEST_SPEC)
        self.assertEqual(set(results), {"base_1x", "base_2x",
                                        "doubled_cost_2x", "no_funding_credit_2x"})
        first = results["base_2x"].trades[0]
        self.assertEqual(first.entry_signal_time, START + timedelta(hours=1))
        self.assertEqual(first.entry_fill_time, START + timedelta(hours=2, minutes=1))
        self.assertEqual(first.exit_fill_time, START + timedelta(hours=5, minutes=1))
        self.assertEqual(first.direction, "short_btc_long_eth")
        self.assertLess(first.funding_pnl, 0.0)
        schedules = [tuple((trade.entry_fill_time, trade.exit_fill_time,
                            trade.direction) for trade in result.trades)
                     for result in results.values()]
        self.assertTrue(all(item == schedules[0] for item in schedules))
        for result in results.values():
            self.assertEqual(len(result.daily_records), 1)
            self.assertEqual(result.completed_cycles, 4)
            day = result.daily_records[0]
            self.assertEqual(day.date, START.date())
            self.assertEqual(day.equity_start, 10_000.0)
            self.assertAlmostEqual(day.equity_end, result.final_equity)
            self.assertGreaterEqual(day.max_regular_factor, 1)
            self.assertLessEqual(day.min_adverse_factor, 1)
            self.assertAlmostEqual(day.max_intraday_drawdown, result.max_drawdown)
        self.assertLess(results["doubled_cost_2x"].final_equity,
                        results["base_2x"].final_equity)
        self.assertLess(results["no_funding_credit_2x"].final_equity,
                        results["base_2x"].final_equity)

    def test_entry_minute_is_stressed_and_exit_minute_is_excluded(self):
        entry_time = START + timedelta(hours=2, minutes=1)
        exit_time = START + timedelta(hours=5, minutes=1)

        def with_spike(at: datetime):
            for bar in synthetic_bars():
                if bar.time == at:
                    yield replace(bar, btc_high=bar.btc_open * 1.10,
                                  eth_low=bar.eth_open * 0.90)
                else:
                    yield bar

        baseline = _simulate(synthetic_bars(), [], spec=TEST_SPEC)["base_2x"]
        entry_spike = _simulate(with_spike(entry_time), [], spec=TEST_SPEC)["base_2x"]
        exit_spike = _simulate(with_spike(exit_time), [], spec=TEST_SPEC)["base_2x"]
        self.assertGreater(entry_spike.max_drawdown, baseline.max_drawdown + 0.10)
        self.assertAlmostEqual(entry_spike.final_equity, baseline.final_equity)
        self.assertAlmostEqual(exit_spike.max_drawdown, baseline.max_drawdown)
        self.assertAlmostEqual(exit_spike.final_equity, baseline.final_equity)

    def test_midnight_funding_zero_and_two_ms_use_prior_close_and_new_day(self):
        spec = _Spec(INPUT_START, START, START + timedelta(days=2), 4, 30, False)
        midnight = START + timedelta(days=1)
        none = _simulate(midnight_bars(), [], spec=spec)["base_2x"]
        outcomes = []
        for offset_ms in (0, 2):
            event = Funding(midnight + timedelta(milliseconds=offset_ms),
                            "BTCUSDT", 0.001)
            outcomes.append(_simulate(midnight_bars(), [event], spec=spec)["base_2x"])
        for result in outcomes:
            self.assertEqual(len(result.trades), 1)
            trade = result.trades[0]
            self.assertLess(trade.entry_fill_time, midnight)
            self.assertGreater(trade.exit_fill_time, midnight)
            self.assertEqual(trade.exit_fill_time, START + timedelta(hours=30, minutes=1))
            prior_close = next(bar.btc_close for bar in midnight_bars()
                               if bar.time == midnight - timedelta(minutes=1))
            self.assertAlmostEqual(trade.funding_pnl,
                                   0.001 * trade.btc_quantity * prior_close)
            self.assertAlmostEqual(result.daily_records[0].equity_end,
                                   none.daily_records[0].equity_end)
            self.assertAlmostEqual(result.daily_records[1].equity_start,
                                   result.daily_records[0].equity_end)
            self.assertGreater(result.daily_records[1].equity_end,
                               none.daily_records[1].equity_end)
        self.assertAlmostEqual(outcomes[0].trades[0].funding_pnl,
                               outcomes[1].trades[0].funding_pnl)

    def test_missing_minute_and_missing_funding_anchor_reject(self):
        bars = list(synthetic_bars())
        del bars[100]
        with self.assertRaises(PairSimulationDataError):
            _simulate(bars, [], spec=TEST_SPEC)
        full_grid_spec = _Spec(INPUT_START, START, END, 4, 5, True)
        with self.assertRaises(PairSimulationDataError):
            _simulate(synthetic_bars(), [], spec=full_grid_spec)


if __name__ == "__main__":
    unittest.main()
