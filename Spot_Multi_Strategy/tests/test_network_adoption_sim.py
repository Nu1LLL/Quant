"""Synthetic-only checks for the frozen network-adoption ledger."""

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import unittest

from research.network_adoption_sim import (
    DataIneligibleError,
    address_growth_score,
    monte_carlo,
    run_backtest,
    run_registered_scenarios,
    summarize_returns,
    yearly_folds,
    _returns_metrics,
)


UTC = timezone.utc
START = datetime(2021, 1, 1, tzinfo=UTC)
END = datetime(2021, 1, 8, tzinfo=UTC)
SYMS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")


@dataclass(frozen=True)
class Bar:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    quote_volume: float | None


def fixtures():
    adrs = {}
    for index, sym in enumerate(SYMS):
        points = {}
        first = date(2020, 11, 25)
        for day_no in range(40):
            day = first + timedelta(days=day_no)
            count = 100.0
            if day >= date(2020, 12, 26):
                count = (200.0, 150.0, 75.0, 50.0)[index]
            points[day] = (count, datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC))
        adrs[sym] = points
    trade = {sym: {} for sym in SYMS}
    mark = {sym: {} for sym in SYMS}
    mark1 = {sym: {} for sym in SYMS}
    funding = {sym: {} for sym in SYMS}
    stamp = datetime(2020, 12, 1, tzinfo=UTC)
    while stamp <= END:
        for sym in SYMS:
            trade[sym][stamp] = Bar(stamp, 100.0, 100.0, 100.0, 100.0, 1_000_000.0)
            mark[sym][stamp] = Bar(stamp, 100.0, 101.0, 99.0, 100.0, None)
            mark1[sym][stamp] = Bar(stamp, 100.0, 100.0, 100.0, 100.0, None)
        stamp += timedelta(hours=4)
    return adrs, trade, mark, mark1, funding


class NetworkAdoptionSimulatorTests(unittest.TestCase):
    def test_signal_uses_prior_friday_and_fails_closed(self):
        adrs, *_ = fixtures()
        decision = datetime(2021, 1, 4, 8, tzinfo=UTC)
        scores = [address_growth_score(adrs[sym], decision) for sym in SYMS]
        self.assertEqual(sorted(scores, reverse=True), scores)
        self.assertGreater(scores[0], 0)
        self.assertLess(scores[-1], 0)
        edited = dict(adrs[SYMS[0]])
        edited[date(2021, 1, 1)] = (999_999.0, datetime(2021, 1, 4, 9, tzinfo=UTC))
        self.assertIsNone(address_growth_score(edited, decision))
        edited[date(2021, 1, 1)] = (999_999.0, datetime(2021, 1, 4, 7, tzinfo=UTC))
        self.assertIsNotNone(address_growth_score(edited, decision))
        edited.pop(date(2020, 12, 20))
        self.assertIsNone(address_growth_score(edited, decision))

    def test_four_hour_delay_costs_funding_daily_calendar(self):
        adrs, trade, mark, mark1, funding = fixtures()
        funding[SYMS[0]][datetime(2021, 1, 5, 8, tzinfo=UTC)] = 0.01
        result = run_backtest(adrs, trade, mark, mark1, funding,
                              start=START, end=END, symbols=SYMS)
        equity = dict(result.daily_equity)
        self.assertEqual(len(result.daily_returns), 7)
        self.assertEqual(equity[date(2021, 1, 3)], 10_000.0)
        self.assertLess(equity[date(2021, 1, 4)], 10_000.0)
        self.assertEqual(result.trade_count, 8)
        self.assertAlmostEqual(result.funding_usdt, -25.0, places=7)
        self.assertGreater(result.fee_usdt, 0)
        self.assertGreater(result.slippage_usdt, 0)
        self.assertEqual(result.cash_days, 3)
        self.assertEqual(result.active_days, 4)
        self.assertFalse(result.independent_oos)
        self.assertLess(summarize_returns(result).cagr, 0)

    def test_flat_market_exact_round_trip_cost_and_daily_pf(self):
        adrs, trade, mark, mark1, funding = fixtures()
        result = run_backtest(adrs, trade, mark, mark1, funding,
                              start=START, end=END, symbols=SYMS)
        equity = dict(result.daily_equity)
        # Four 2,500-USDT legs are opened and closed at unchanged trade and
        # mark prices.  The only possible loss is 15 bp adverse fills and
        # 5 bp taker fees on each of the two 10,000-USDT turnovers.
        self.assertAlmostEqual(equity[date(2021, 1, 4)], 9_980.0)
        self.assertAlmostEqual(equity[date(2021, 1, 7)], 9_960.0)
        self.assertAlmostEqual(result.turnover_usdt, 20_000.0)
        self.assertAlmostEqual(result.slippage_usdt, 30.0)
        self.assertAlmostEqual(result.fee_usdt, 10.0)
        self.assertAlmostEqual(result.funding_usdt, 0.0)
        self.assertAlmostEqual(10_000.0 - equity[date(2021, 1, 7)],
                               result.fee_usdt + result.slippage_usdt)
        self.assertEqual(summarize_returns(result).profit_factor, 0.0)

    def test_funding_at_entry_and_rebalance_uses_old_position(self):
        adrs, trade, mark, mark1, funding = fixtures()
        end = datetime(2021, 1, 15, tzinfo=UTC)
        stamp = END + timedelta(hours=4)
        while stamp <= end:
            for sym in SYMS:
                trade[sym][stamp] = Bar(stamp, 100.0, 100.0, 100.0, 100.0, 1_000_000.0)
                mark[sym][stamp] = Bar(stamp, 100.0, 101.0, 99.0, 100.0, None)
                mark1[sym][stamp] = Bar(stamp, 100.0, 100.0, 100.0, 100.0, None)
            stamp += timedelta(hours=4)
        for day_no in range(5):
            day = date(2021, 1, 4) + timedelta(days=day_no)
            for sym in SYMS:
                count = 1.0 if sym == SYMS[0] else 1000.0 if sym == SYMS[3] else 100.0
                adrs[sym][day] = (
                    count, datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC)
                )
        # At the first Monday 12:00 there is no old position; at the second
        # Monday the formerly long first symbol becomes a short.  Funding at
        # that second timestamp must be charged on the *old long* first.
        entry = datetime(2021, 1, 4, 12, tzinfo=UTC)
        rebalance = datetime(2021, 1, 11, 12, tzinfo=UTC)
        funding[SYMS[0]][entry] = 0.01
        funding[SYMS[0]][rebalance] = 0.01
        result = run_backtest(adrs, trade, mark, mark1, funding,
                              start=START, end=end, symbols=SYMS)
        self.assertAlmostEqual(result.funding_usdt, -25.0)

    def test_active_funding_without_exact_hourly_mark_is_data_error(self):
        adrs, trade, mark, mark1, funding = fixtures()
        settlement = datetime(2021, 1, 5, 8, tzinfo=UTC)
        funding[SYMS[0]][settlement + timedelta(milliseconds=33)] = 0.01
        del mark1[SYMS[0]][settlement]
        with self.assertRaisesRegex(DataIneligibleError, "missing settlement mark1"):
            run_backtest(adrs, trade, mark, mark1, funding,
                         start=START, end=END, symbols=SYMS)

    def test_missing_unannounced_market_bar_is_data_error(self):
        adrs, trade, mark, mark1, funding = fixtures()
        del trade[SYMS[3]][datetime(2021, 1, 2, 4, tzinfo=UTC)]
        with self.assertRaises(DataIneligibleError):
            run_backtest(adrs, trade, mark, mark1, funding,
                         start=START, end=END, symbols=SYMS)

    def test_funding_millisecond_tolerance_only_when_held(self):
        adrs, trade, mark, mark1, funding = fixtures()
        funding[SYMS[0]][datetime(2021, 1, 2, 8, 0, 0, 101_000, UTC)] = 0.01
        run_backtest(adrs, trade, mark, mark1, funding,
                     start=START, end=END, symbols=SYMS)
        funding[SYMS[0]][datetime(2021, 1, 5, 8, 0, 0, 10_000, UTC)] = 0.01
        result = run_backtest(adrs, trade, mark, mark1, funding,
                              start=START, end=END, symbols=SYMS)
        self.assertAlmostEqual(result.funding_usdt, -25.0)
        funding[SYMS[0]][datetime(2021, 1, 5, 8, 0, 0, 101_000, UTC)] = 0.01
        with self.assertRaises(DataIneligibleError):
            run_backtest(adrs, trade, mark, mark1, funding,
                         start=START, end=END, symbols=SYMS)

    def test_duplicate_same_logical_funding_hour_is_data_error(self):
        adrs, trade, mark, mark1, funding = fixtures()
        funding[SYMS[0]][datetime(2021, 1, 5, 8, 0, 0, 10_000, UTC)] = 0.01
        funding[SYMS[0]][datetime(2021, 1, 5, 8, 0, 0, 20_000, UTC)] = 0.01
        with self.assertRaises(DataIneligibleError):
            run_backtest(adrs, trade, mark, mark1, funding,
                         start=START, end=END, symbols=SYMS)

    def test_terminal_funding_is_paid_before_exit(self):
        adrs, trade, mark, mark1, funding = fixtures()
        baseline = run_backtest(adrs, trade, mark, mark1, funding,
                                start=START, end=END, symbols=SYMS)
        funding[SYMS[0]][datetime(2021, 1, 8, 0, 0, 0, 33_000, UTC)] = 0.01
        with_funding = run_backtest(adrs, trade, mark, mark1, funding,
                                    start=START, end=END, symbols=SYMS)
        self.assertAlmostEqual(with_funding.funding_usdt, -25.0)
        self.assertAlmostEqual(baseline.daily_equity[-1][1] - with_funding.daily_equity[-1][1], 25.0)

    def test_intrabar_funding_does_not_leak_to_previous_utc_day(self):
        adrs, trade, mark, mark1, funding = fixtures()
        baseline = run_backtest(adrs, trade, mark, mark1, funding,
                                start=START, end=END, symbols=SYMS)
        logical = datetime(2021, 1, 5, 1, tzinfo=UTC)
        mark1[SYMS[0]][logical] = Bar(logical, 100.0, 100.0, 100.0, 100.0, None)
        funding[SYMS[0]][logical + timedelta(milliseconds=10)] = 0.01
        with_funding = run_backtest(adrs, trade, mark, mark1, funding,
                                    start=START, end=END, symbols=SYMS)
        base_equity = dict(baseline.daily_equity)
        funded_equity = dict(with_funding.daily_equity)
        self.assertAlmostEqual(base_equity[date(2021, 1, 4)], funded_equity[date(2021, 1, 4)])
        self.assertAlmostEqual(base_equity[date(2021, 1, 5)] - funded_equity[date(2021, 1, 5)], 25.0)

    def test_notice_closes_every_leg_and_no_same_week_reentry(self):
        adrs, trade, mark, mark1, funding = fixtures()
        notice = {SYMS[0]: datetime(2021, 1, 5, 9, tzinfo=UTC)}
        result = run_backtest(adrs, trade, mark, mark1, funding,
                              announcements=notice, start=START, end=END, symbols=SYMS)
        self.assertEqual(result.trade_count, 8)
        self.assertEqual(result.active_days, 2)
        self.assertEqual(result.cash_days, 5)

    def test_notice_exact_decision_and_execution_boundaries(self):
        adrs, trade, mark, mark1, funding = fixtures()
        decision = datetime(2021, 1, 4, 8, tzinfo=UTC)
        execution = datetime(2021, 1, 4, 12, tzinfo=UTC)
        before_decision = run_backtest(
            adrs, trade, mark, mark1, funding,
            announcements={SYMS[0]: decision - timedelta(microseconds=1)},
            start=START, end=END, symbols=SYMS,
        )
        between = run_backtest(
            adrs, trade, mark, mark1, funding,
            announcements={SYMS[0]: execution - timedelta(microseconds=1)},
            start=START, end=END, symbols=SYMS,
        )
        at_decision = run_backtest(
            adrs, trade, mark, mark1, funding,
            announcements={SYMS[0]: decision},
            start=START, end=END, symbols=SYMS,
        )
        at_execution = run_backtest(
            adrs, trade, mark, mark1, funding,
            announcements={SYMS[0]: execution},
            start=START, end=END, symbols=SYMS,
        )
        self.assertEqual(before_decision.trade_count, 0)
        self.assertEqual(between.trade_count, 0)
        self.assertEqual(at_decision.trade_count, 0)  # known during four-hour order delay
        self.assertEqual(at_execution.trade_count, 8)

    def test_notice_full_exit_supersedes_same_open_leverage_cut(self):
        adrs, trade, mark, mark1, funding = fixtures()
        for hour in (8, 12):
            stamp = datetime(2021, 1, 5, hour, tzinfo=UTC)
            for sym in SYMS[:2]:
                mark[sym][stamp] = Bar(stamp, 100.0, 100.0, 100.0, 100.0, None)
            for sym in SYMS[2:]:
                mark[sym][stamp] = Bar(stamp, 295.0, 295.0, 295.0, 295.0, None)
        result = run_backtest(
            adrs, trade, mark, mark1, funding,
            announcements={SYMS[0]: datetime(2021, 1, 5, 9, tzinfo=UTC)},
            start=START, end=END, symbols=SYMS,
        )
        self.assertTrue(result.leverage_failed)
        self.assertFalse(result.liquidated)
        self.assertEqual(result.trade_count, 8)  # four entry, four notice-exit; no partial cut

    def test_capacity_check_does_not_shrink_orders(self):
        adrs, trade, mark, mark1, funding = fixtures()
        result = run_backtest(adrs, trade, mark, mark1, funding,
                              start=START, end=END, symbols=SYMS, initial_equity=10_000_000.0)
        self.assertTrue(result.capacity_failed)
        self.assertGreater(result.turnover_usdt, 10_000_000.0)

    def test_adverse_mark_liquidates_and_never_recovers(self):
        adrs, trade, mark, mark1, funding = fixtures()
        time = datetime(2021, 1, 5, 0, tzinfo=UTC)
        for sym in SYMS:
            mark[sym][time] = Bar(time, 100.0, 200.0, 0.1, 100.0, None)
        result = run_backtest(adrs, trade, mark, mark1, funding,
                              start=START, end=END, symbols=SYMS)
        self.assertTrue(result.liquidated)
        self.assertEqual(result.daily_equity[-1][1], 0.0)
        self.assertEqual(result.daily_returns[-1][1], 0.0)
        self.assertEqual(result.stress_drawdown, -1.0)

    def test_no_3x_5x_when_1x_quality_fails(self):
        adrs, trade, mark, mark1, funding = fixtures()
        outcomes = run_registered_scenarios(adrs, trade, mark, mark1, funding,
                                            start=START, end=END, symbols=SYMS)
        self.assertEqual(set(outcomes), {"1x_15bp", "1x_25bp_stress"})

    def test_annual_fold_includes_next_midnight_boundary_equity(self):
        adrs, trade, mark, mark1, funding = fixtures()
        template = run_backtest(adrs, trade, mark, mark1, funding,
                                start=START, end=END, symbols=SYMS)
        days = [date(2021, 1, 1) + timedelta(days=offset) for offset in range(365)]
        fake = replace(
            template,
            start=datetime(2021, 1, 1, tzinfo=UTC),
            end=datetime(2022, 1, 1, tzinfo=UTC),
            daily_returns=tuple((day, -0.5 if day == days[-1] else 0.0) for day in days),
            daily_equity=tuple((day, 5_000.0 if day == days[-1] else 10_000.0) for day in days),
            stress_points=(),
        )
        folds = yearly_folds(fake)
        self.assertEqual(len(folds), 1)
        self.assertAlmostEqual(folds[0].stress_drawdown, -0.5)

    def test_block_mc_is_seeded_and_returns_joint_probability(self):
        values = [0.01, -0.008, 0.0, 0.003] * 9
        first = monte_carlo(values, paths=50, block_days=3, seed=9)
        second = monte_carlo(values, paths=50, block_days=3, seed=9)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.joint_pass_share, 0)
        self.assertLessEqual(first.joint_pass_share, 1)

    def test_tiny_nonzero_sample_variance_has_sharpe(self):
        metric = _returns_metrics([1e-12, 2e-12, 3e-12, 4e-12])
        self.assertIsNotNone(metric.sharpe)
        self.assertGreater(metric.sharpe, 0)
        self.assertIsNone(_returns_metrics([1e-12] * 4).sharpe)


if __name__ == "__main__":
    unittest.main()
