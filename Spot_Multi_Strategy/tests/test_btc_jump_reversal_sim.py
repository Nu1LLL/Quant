"""Synthetic causal and accounting checks for the frozen BTC experiment."""

import math
import unittest

from research.btc_jump_reversal_sim import (
    ADVERSE_RATE, DAY_MS, FEE_RATE, INITIAL_EQUITY, MINUTE_MS,
    OOS_START_MS, TWO_HOURS_MS, WARMUP_START_MS, DataGateError, FundingRate, MinuteBar,
    TwoHourSignal, _Scenario, _daily_returns, _path_metrics, monte_carlo,
    simulate, StrategyFailure,
)


def bar(time_ms, price=100, *, high=None, low=None):
    return MinuteBar(
        time_ms, price, price if high is None else high,
        price if low is None else low, price,
    )


class SignalTests(unittest.TestCase):
    def test_current_jump_is_excluded_from_sigma_and_reversed(self):
        state = TwoHourSignal()
        close = 100.0
        boundary = OOS_START_MS - 4002 * TWO_HOURS_MS
        for index in range(4001):
            close *= math.exp(0.001 if index % 2 == 0 else -0.001)
            direction, _ = state.observe(boundary, close)
            self.assertEqual(direction, 0)
            boundary += TWO_HOURS_MS
        close *= math.exp(0.005)
        direction, regime = state.observe(boundary, close)
        self.assertEqual(direction, -1)
        self.assertIn(regime, ("bull_high", "bull_low", "bear_high", "bear_low"))

    def test_invalid_price_and_grid_rejected(self):
        with self.assertRaises(DataGateError):
            bar(OOS_START_MS + 1)
        with self.assertRaises(DataGateError):
            MinuteBar(OOS_START_MS, 100, 99, 101, 100)
        with self.assertRaises(DataGateError):
            FundingRate(OOS_START_MS, float("nan"))

    def test_same_day_midnight_vol_snapshot_is_excluded_at_0200(self):
        state = TwoHourSignal()
        for index in range(360):
            state.returns.append(1.5 if index % 2 else -1.5)
        state.previous_close = 100
        state.past_closes.extend([100] * 1080)
        today = OOS_START_MS // DAY_MS
        for index in range(300):
            state.daily_volatility.append((today - 300 + index,
                                           1.0 if index < 150 else 3.0))
        state.daily_volatility.append((today, 0.0))
        direction, regime = state.observe(OOS_START_MS + TWO_HOURS_MS, 100)
        self.assertEqual(direction, 0)
        self.assertEqual(regime, "bull_low")


class CashAndCostTests(unittest.TestCase):
    def test_round_trip_fee_adverse_fill_and_funding(self):
        scenario = _Scenario(ADVERSE_RATE, False)
        start = OOS_START_MS
        scenario.enter(bar(start), 1, start + 120 * MINUTE_MS, "bull_high")
        position = scenario.position
        self.assertAlmostEqual(position.entry_fill, 100 * (1 + ADVERSE_RATE))
        self.assertAlmostEqual(position.quantity * position.entry_fill, INITIAL_EQUITY)
        self.assertAlmostEqual(scenario.cash, INITIAL_EQUITY * (1 - FEE_RATE))
        scenario.funding(0.001, 100)
        self.assertLess(scenario.cash, INITIAL_EQUITY * (1 - FEE_RATE))
        expected_funding = -0.001 * position.quantity * 100
        scenario.exit(bar(start + 120 * MINUTE_MS))
        trade = scenario.trades[0]
        self.assertEqual(trade.funding_pnl, expected_funding)
        self.assertLess(trade.net_pnl, -39)
        self.assertEqual(scenario.position, None)

    def test_funding_credit_suppression_and_short_adverse_high(self):
        start = OOS_START_MS
        scenario = _Scenario(ADVERSE_RATE, True)
        scenario.enter(bar(start), -1, start + 120 * MINUTE_MS, "bear_low")
        before = scenario.cash
        scenario.funding(0.001, 100)  # Beneficial to a short, suppressed.
        self.assertEqual(scenario.cash, before)
        scenario.mark(bar(start + MINUTE_MS, high=120, low=90))
        self.assertGreater(scenario.max_drawdown, 0.19)
        with self.assertRaises(StrategyFailure):
            scenario.mark(bar(start + 2 * MINUTE_MS, high=220, low=90))

    def test_dollar_profit_factor_recompounds(self):
        metrics = _path_metrics([0.1, -0.05], initial=100)
        self.assertAlmostEqual(metrics["profit_factor"], 10 / 5.5)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.05)
        self.assertAlmostEqual(_daily_returns([110, 104.5], start=100)[1], -0.05)

    def test_block_bootstrap_is_deterministic(self):
        returns = [0.01, -0.01, 0.02, 0.0] * 10
        first = monte_carlo(returns, paths=30, seed=7, block_days=3)
        second = monte_carlo(returns, paths=30, seed=7, block_days=3)
        self.assertEqual(first, second)
        self.assertEqual(first["paths"], 30)
        self.assertEqual(first["distributions"]["cagr"]["undefined_count"], 0)

    def test_partial_history_rejected_without_partial_result(self):
        with self.assertRaisesRegex(DataGateError, "ends before complete"):
            simulate([bar(WARMUP_START_MS)], [])


if __name__ == "__main__":
    unittest.main()
