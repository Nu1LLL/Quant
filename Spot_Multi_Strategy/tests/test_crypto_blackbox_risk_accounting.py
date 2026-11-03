"""Synthetic-only contracts for the isolated crypto PAPER account."""

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from research.crypto_blackbox.risk_accounting import (
    AccountConfig,
    ExternalFlow,
    Fill,
    Funding,
    InternalTransfer,
    Mark,
    OrderIntent,
    PaperAccount,
)


D = Decimal
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def intent(
    order_id: str,
    at: datetime,
    *,
    symbol: str = "BTCUSDT",
    bucket: str = "core",
    cluster: str = "ALT",
    side: str = "BUY",
    reference: str = "100",
    baseline: str = "1000",
    p: str = "1",
    odds: str = "1",
    haircut: str = "1",
    leverage: str = "5",
    instrument_type: str = "CRYPTO_LINEAR_USDT_PERP",
    venue: str = "BINANCE_FUTURES",
) -> OrderIntent:
    return OrderIntent(
        order_id, at, symbol, bucket, cluster, side, D(reference), D(baseline),
        D(p), D(odds), D(haircut), D(leverage), instrument_type, venue,
    )


class RiskContractTests(unittest.TestCase):
    def test_three_buckets_default_reserve_never_opens(self):
        account = PaperAccount(AccountConfig(), T0)
        snap = account.snapshot()
        self.assertEqual(dict(snap.bucket_cash), {
            "core": D("3300"), "tactical": D("3300"), "reserve": D("3400")
        })
        self.assertEqual(snap.equity, D("10000"))
        self.assertTrue(snap.paper_only)
        decision = account.assess_order(intent("r1", T0, bucket="reserve"))
        self.assertEqual(decision.allowed_notional, D("0"))
        self.assertEqual(decision.reason, "reserve_bucket_no_new_entries")

    def test_kelly_is_discounted_and_never_exceeds_baseline(self):
        account = PaperAccount(AccountConfig(), T0)
        i = intent("k1", T0, baseline="1000", p="0.6", odds="1", haircut="0.5")
        decision = account.assess_order(i)
        self.assertEqual(decision.kelly_fraction, D("0.025"))
        self.assertEqual(decision.allowed_notional, D("1000"))
        low = account.assess_order(intent("k2", T0, p="0.49", odds="1"))
        self.assertEqual(low.allowed_notional, D("0"))
        capped = account.assess_order(intent("k3", T0, baseline="20000", p="0.6", odds="1"))
        self.assertEqual(capped.allowed_notional, D("2500.00"))

    def test_leverage_and_invalid_forecast_rejected(self):
        with self.assertRaisesRegex(ValueError, "above 5x"):
            intent("10x", T0, leverage="10")
        with self.assertRaisesRegex(ValueError, "above 5x"):
            intent("20x", T0, leverage="20")
        with self.assertRaisesRegex(ValueError, "p_win_lower"):
            intent("bad-p", T0, p="1.01")
        with self.assertRaisesRegex(TypeError, "finite Decimal"):
            AccountConfig(initial_equity=10000.0)
        with self.assertRaisesRegex(ValueError, "only CRYPTO_LINEAR_USDT_PERP"):
            intent("stock", T0, symbol="AAPL", instrument_type="EQUITY")
        with self.assertRaisesRegex(ValueError, "nonempty venue"):
            intent("no-venue", T0, venue=" ")
        with self.assertRaisesRegex(ValueError, "max_drawdown_mark_gap"):
            AccountConfig(max_drawdown_mark_gap=timedelta(hours=2))

    def test_symbol_cluster_bucket_and_global_caps(self):
        config = AccountConfig(fractional_kelly=D("1"))
        account = PaperAccount(config, T0)
        first = intent("c1", T0, baseline="20000")
        self.assertEqual(account.assess_order(first).allowed_notional, D("10000"))
        account.record_fill(Fill(T0 + timedelta(seconds=1), first, D("100"), D("100"), D("0")))
        second = intent("c2", T0 + timedelta(seconds=2), symbol="ETHUSDT", baseline="20000")
        self.assertEqual(account.assess_order(second).allowed_notional, D("6500"))
        # Extra names supplied by a caller cannot evade the conservative ALT cluster.
        with self.assertRaisesRegex(ValueError, "canonical"):
            account.assess_order(intent("fake-cluster", T0 + timedelta(seconds=2), cluster="OTHER"))

    def test_stale_other_symbol_mark_blocks_new_risk_until_refreshed(self):
        account = PaperAccount(AccountConfig(fractional_kelly=D("1")), T0)
        opening = intent("m1", T0, baseline="1000")
        account.record_fill(Fill(T0 + timedelta(seconds=1), opening, D("10"), D("100"), D("0")))
        later = T0 + timedelta(days=1)
        candidate = intent("m2", later, symbol="ETHUSDT", baseline="1000")
        stale = account.assess_order(candidate)
        self.assertEqual(stale.allowed_notional, D("0"))
        self.assertEqual(stale.reason, "stale_valuation_no_new_entries")
        account.record_mark(Mark(later, "BTCUSDT", D("100")))
        refreshed = account.assess_order(candidate)
        self.assertEqual(refreshed.allowed_notional, D("1000"))

    def test_existing_symbol_cannot_be_net_closed_on_another_venue(self):
        account = PaperAccount(AccountConfig(fractional_kelly=D("1")), T0)
        opening = intent("v1", T0, baseline="1000")
        account.record_fill(Fill(T0 + timedelta(seconds=1), opening, D("10"), D("100"), D("0")))
        wrong_venue = intent("v2", T0 + timedelta(seconds=2), side="SELL",
                             baseline="1000", p="0", venue="OTHER_FUTURES")
        with self.assertRaisesRegex(ValueError, "another venue"):
            account.assess_order(wrong_venue)

    def test_cannot_fill_more_than_baseline_or_approved_notional(self):
        account = PaperAccount(AccountConfig(), T0)
        tiny_edge = intent("tiny", T0, baseline="10000", p="0.6", odds="1")
        self.assertEqual(account.assess_order(tiny_edge).allowed_notional, D("2500"))
        with self.assertRaisesRegex(ValueError, "risk approval"):
            account.record_fill(Fill(T0 + timedelta(seconds=1), tiny_edge, D("30"), D("100"), D("1")))
        with self.assertRaisesRegex(ValueError, "baseline"):
            account.record_fill(Fill(T0 + timedelta(seconds=1), tiny_edge, D("101"), D("100"), D("1")))
        self.assertEqual(account.snapshot().equity, D("10000"))
        self.assertEqual(len(account.ledger), 0)


class AccountingContractTests(unittest.TestCase):
    def setUp(self):
        self.account = PaperAccount(AccountConfig(fractional_kelly=D("1")), T0)

    def test_fifo_partial_close_flip_short_and_funding(self):
        a = intent("a", T0, baseline="1000")
        self.account.record_fill(Fill(T0 + timedelta(seconds=1), a, D("10"), D("100"), D("2")))
        self.account.record_funding(Funding(T0 + timedelta(hours=1), "BTCUSDT", "core", D("1")))
        b = intent("b", T0 + timedelta(hours=2), baseline="1000", reference="110")
        self.account.record_fill(Fill(T0 + timedelta(hours=2, seconds=1), b, D("5"), D("110"), D("1")))
        short = intent("c", T0 + timedelta(hours=3), side="SELL", baseline="2200", reference="120")
        self.account.record_fill(Fill(T0 + timedelta(hours=3, seconds=1), short, D("15"), D("120"), D("3")))
        snap = self.account.snapshot()
        # FIFO: close 10 @100 and 5 @110 at 120, then open 0: 200 + 50 gross.
        self.assertEqual(snap.realized_price_pnl, D("250"))
        self.assertEqual(snap.commission_paid, D("6"))
        self.assertEqual(snap.funding_net, D("1"))
        self.assertEqual(snap.cash, D("10245"))
        self.assertEqual(snap.unrealized_pnl, D("0"))
        self.assertEqual(len(self.account.closed_trades), 2)
        self.assertEqual(self.account.closed_trades[0].net_pnl, D("197"))
        self.assertEqual(self.account.closed_trades[1].net_pnl, D("48"))
        self.assertEqual(sum(t.net_pnl for t in self.account.closed_trades), D("245"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.account.record_fill(Fill(T0 + timedelta(hours=4), short, D("1"), D("120"), D("0")))

    def test_short_profit_and_mark_to_market_equity(self):
        sell = intent("s1", T0, side="SELL", baseline="1000")
        self.account.record_fill(Fill(T0 + timedelta(seconds=1), sell, D("10"), D("100"), D("1")))
        self.account.record_mark(Mark(T0 + timedelta(days=1), "BTCUSDT", D("90")))
        self.assertEqual(self.account.snapshot().equity, D("10099"))
        buy = intent("s2", T0 + timedelta(days=2), side="BUY", reference="80", baseline="800", p="0")
        self.account.record_fill(Fill(T0 + timedelta(days=2, seconds=1), buy, D("10"), D("80"), D("1")))
        snap = self.account.snapshot()
        self.assertEqual(snap.realized_price_pnl, D("200"))
        self.assertEqual(snap.equity, D("10198"))
        self.assertEqual(snap.gross_notional, D("0"))
        self.assertEqual(self.account.closed_trades[0].net_pnl, D("198"))

    def test_partial_closes_allocate_entry_fee_and_funding_once(self):
        opening = intent("p1", T0, baseline="1000")
        self.account.record_fill(Fill(T0 + timedelta(seconds=1), opening, D("10"), D("100"), D("2")))
        self.account.record_funding(Funding(T0 + timedelta(hours=1), "BTCUSDT", "core", D("1")))
        first = intent("p2", T0 + timedelta(hours=2), side="SELL", reference="110",
                       baseline="550", p="0")
        self.account.record_fill(Fill(T0 + timedelta(hours=2, seconds=1), first, D("5"), D("110"), D("1")))
        self.assertEqual(self.account.closed_trades[0].net_pnl, D("48.5"))
        second = intent("p3", T0 + timedelta(hours=3), side="SELL", reference="120",
                        baseline="600", p="0")
        self.account.record_fill(Fill(T0 + timedelta(hours=3, seconds=1), second, D("5"), D("120"), D("1")))
        self.assertEqual(self.account.closed_trades[1].net_pnl, D("98.5"))
        self.assertEqual(self.account.snapshot().equity, D("10147"))
        self.assertEqual(sum(t.net_pnl for t in self.account.closed_trades), D("147"))

    def test_external_deposit_and_transfer_cannot_fake_return(self):
        at = T0 + timedelta(days=30)
        self.account.record_external_flow(ExternalFlow(T0 + timedelta(days=10), "core", D("1000")))
        self.account.record_external_flow(ExternalFlow(T0 + timedelta(days=11), "core", D("-500")))
        self.account.record_internal_transfer(InternalTransfer(T0 + timedelta(days=12), "reserve", "tactical", D("100")))
        self.account.record_mark(Mark(at, "BTCUSDT", D("100")))
        report = self.account.report_30d(at)
        self.assertTrue(report.covered_30d)
        self.assertEqual(report.external_flows_net, D("500"))
        self.assertEqual(report.time_weighted_return, D("0"))
        self.assertEqual(report.profit_ex_external_flows, D("0"))
        self.assertEqual(self.account.snapshot().equity, D("10500"))
        self.assertEqual(self.account.snapshot().max_drawdown, D("0"))

    def test_30d_report_requires_exact_boundary_and_facts(self):
        buy = intent("r1", T0, baseline="1000")
        self.account.record_fill(Fill(T0 + timedelta(hours=1), buy, D("10"), D("100"), D("1")))
        at = T0 + timedelta(days=31)
        self.account.record_mark(Mark(at, "BTCUSDT", D("110")))
        incomplete = self.account.report_30d(at)
        self.assertFalse(incomplete.covered_30d)
        self.assertIsNone(incomplete.time_weighted_return)
        self.assertIsNone(incomplete.win_rate)
        with self.assertRaisesRegex(ValueError, "latest"):
            self.account.report_30d(at - timedelta(seconds=1))

    def test_30d_report_rejects_stale_open_position_mark(self):
        buy = intent("stale", T0, baseline="1000")
        self.account.record_fill(Fill(T0 + timedelta(seconds=1), buy, D("10"), D("100"), D("0")))
        at = T0 + timedelta(days=30)
        self.account.record_external_flow(ExternalFlow(at, "tactical", D("100")))
        self.assertFalse(self.account.snapshot().valuation_fresh)
        report = self.account.report_30d(at)
        self.assertFalse(report.covered_30d)
        self.assertEqual(report.reason, "stale_open_position_mark")
        self.assertIsNone(report.unrealized_pnl_now)

    def test_funded_profitable_30d_report_is_flow_adjusted(self):
        buy = intent("w1", T0, baseline="1000")
        self.account.record_fill(Fill(T0 + timedelta(hours=1), buy, D("10"), D("100"), D("1")))
        self.account.record_external_flow(ExternalFlow(T0 + timedelta(days=10), "core", D("2000")))
        close = intent("w2", T0 + timedelta(days=20), side="SELL", reference="110", baseline="1100", p="0")
        self.account.record_fill(Fill(T0 + timedelta(days=20, seconds=1), close, D("10"), D("110"), D("1")))
        at = T0 + timedelta(days=30)
        self.account.record_mark(Mark(at, "BTCUSDT", D("110")))
        report = self.account.report_30d(at)
        self.assertEqual(report.profit_ex_external_flows, D("98"))
        self.assertEqual(report.realized_net_pnl, D("98"))
        self.assertEqual(report.closed_trade_count, 1)
        self.assertEqual(report.win_rate, D("1"))
        self.assertEqual(report.traded_notional, D("2100"))
        self.assertEqual(report.unrealized_pnl_now, D("0"))
        self.assertEqual(report.external_flows_net, D("2000"))
        self.assertGreater(report.time_weighted_return, D("0"))
        self.assertLess(report.time_weighted_return, D("0.01"))
        # First fill is one hour after its signal; the second is one second.
        self.assertEqual(report.mean_latency_ms, D("1800500"))
        self.assertIsNone(report.max_drawdown)
        self.assertEqual(report.reason, "sparse_intraperiod_marks_drawdown_unknown")

    def test_hourly_full_book_marks_allow_only_sampled_drawdown(self):
        opened_at = T0 + timedelta(days=30, hours=-2)
        buy = intent("hourly", opened_at, baseline="1000")
        self.account.record_fill(Fill(opened_at + timedelta(seconds=1), buy,
                                      D("10"), D("100"), D("1")))
        self.account.record_mark(Mark(opened_at + timedelta(hours=1), "BTCUSDT", D("90")))
        at = T0 + timedelta(days=30)
        self.account.record_mark(Mark(at, "BTCUSDT", D("100")))
        report = self.account.report_30d(at)
        self.assertTrue(report.covered_30d)
        self.assertEqual(report.max_drawdown, D("0.0101"))
        self.assertEqual(report.reason, "ok_hourly_sampled_drawdown")

    def test_drawdown_20_percent_latches_stop_but_allows_exit(self):
        buy = intent("d1", T0, baseline="10000")
        self.account.record_fill(Fill(T0 + timedelta(seconds=1), buy, D("100"), D("100"), D("0")))
        self.account.record_mark(Mark(T0 + timedelta(days=1), "BTCUSDT", D("80")))
        self.assertEqual(self.account.snapshot().current_drawdown, D("0.2"))
        self.assertTrue(self.account.snapshot().new_entries_halted)
        new_entry = intent("d2", T0 + timedelta(days=1, seconds=1), symbol="ETHUSDT")
        self.assertEqual(self.account.assess_order(new_entry).allowed_notional, D("0"))
        exit_intent = intent("d3", T0 + timedelta(days=1, seconds=1), side="SELL",
                             reference="80", baseline="8000", p="0")
        self.assertEqual(self.account.assess_order(exit_intent).allowed_notional, D("8000"))
        self.account.record_fill(Fill(T0 + timedelta(days=1, seconds=2), exit_intent,
                                      D("100"), D("80"), D("0")))
        self.assertEqual(self.account.snapshot().equity, D("8000"))
        self.assertTrue(self.account.snapshot().new_entries_halted)

    def test_future_out_of_order_negative_equity_and_atomic_rollback(self):
        self.account.record_mark(Mark(T0 + timedelta(hours=2), "BTCUSDT", D("100")))
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            self.account.record_mark(Mark(T0 + timedelta(hours=1), "BTCUSDT", D("99")))
        with self.assertRaisesRegex(ValueError, "future"):
            self.account.record_mark(Mark(datetime.now(timezone.utc) + timedelta(days=1),
                                          "BTCUSDT", D("99")))
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            Mark(datetime(2026, 1, 1), "BTCUSDT", D("100"))
        aggressive = PaperAccount(AccountConfig(fractional_kelly=D("1"),
                                                max_symbol_leverage=D("5"),
                                                max_cluster_leverage=D("5")), T0)
        buy = intent("n1", T0, baseline="15000")
        aggressive.record_fill(Fill(T0 + timedelta(seconds=1), buy, D("150"), D("100"), D("0")))
        before = aggressive.snapshot()
        with self.assertRaisesRegex(ValueError, "nonpositive"):
            aggressive.record_mark(Mark(T0 + timedelta(hours=1), "BTCUSDT", D("20")))
        self.assertEqual(aggressive.snapshot(), before)
        self.assertEqual(len(aggressive.equity_points), 2)


if __name__ == "__main__":
    unittest.main()
