"""Synthetic point-in-time five-minute preprocessing tests."""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from research.crypto_blackbox.preprocessing import (
    CompletedTradeWindow,
    OpenInterestSnapshot,
    TradeEvent,
    binance_aggressor_side,
    build_flow_observation,
)


END = datetime(2026, 2, 1, 0, 5, tzinfo=timezone.utc)
META = dict(
    source="synthetic", source_version="v1", license_id="test-only",
    market="crypto_perpetual", venue="BINANCE", symbol="BTCUSDT",
)


def fixture():
    windows = []
    for offset in range(12, -1, -1):
        end = END - timedelta(minutes=5 * offset)
        windows.append(CompletedTradeWindow(
            **META, window_start=end - timedelta(minutes=5), window_end=end,
            available_time=end + timedelta(seconds=1),
            trade_count=2 if offset == 0 else 1,
            quote_volume_usd=100.0, complete=True,
        ))
    trades = [
        TradeEvent(
            **META, event_id="buy-1", event_time=END - timedelta(minutes=4),
            available_time=END - timedelta(minutes=4) + timedelta(milliseconds=100),
            price=10.0, quantity=6.0, aggressor_side="BUY",
        ),
        TradeEvent(
            **META, event_id="sell-1", event_time=END - timedelta(minutes=2),
            available_time=END - timedelta(minutes=2) + timedelta(milliseconds=100),
            price=10.0, quantity=4.0, aggressor_side="SELL",
        ),
    ]
    return trades, windows


class CryptoPreprocessingTests(unittest.TestCase):
    def test_binance_aggressor_definition_is_explicit(self):
        self.assertEqual(binance_aggressor_side(True), "SELL")
        self.assertEqual(binance_aggressor_side(False), "BUY")
        with self.assertRaises(TypeError):
            binance_aggressor_side(1)
        with self.assertRaises(ValueError):
            TradeEvent(**META, event_id="x", event_time=END - timedelta(minutes=1),
                       available_time=END, price=1, quantity=1, aggressor_side="UNKNOWN")

    def test_current_trade_share_and_prior_twelve_window_mean(self):
        trades, windows = fixture()
        result = build_flow_observation(
            trades, windows, decision_time=END + timedelta(seconds=2),
        )
        self.assertEqual(result.family, "flow_volume_position")
        self.assertEqual(result.window_start, END - timedelta(minutes=5))
        self.assertEqual(result.event_time, END)
        self.assertEqual(result.available_time, END + timedelta(seconds=1))
        self.assertAlmostEqual(result.observed_fields["aggressor_buy_share_5m"], 0.6)
        self.assertAlmostEqual(result.observed_fields["quote_volume_5m_usd"], 100)
        self.assertAlmostEqual(result.observed_fields["quote_volume_baseline_5m_usd"], 100)
        self.assertEqual(result.observed_fields["trade_count_5m"], 2)
        self.assertEqual(result.license_id, "test-only")
        self.assertEqual(result.event_id, build_flow_observation(
            trades, windows, decision_time=END + timedelta(seconds=3),
        ).event_id)

    def test_missing_duplicate_or_unfinished_window_fails_closed(self):
        trades, windows = fixture()
        for name, changed in (
            ("missing", windows[1:]),
            ("duplicate", windows[:-1] + [windows[-2]]),
        ):
            with self.subTest(case=name), self.assertRaises(ValueError):
                build_flow_observation(trades, changed, decision_time=END + timedelta(seconds=2))
        with self.assertRaises(ValueError):
            replace(windows[0], complete=False)
        with self.assertRaises(ValueError):
            replace(windows[0], available_time=windows[0].window_end - timedelta(seconds=1))
        gap = windows[:3] + windows[4:] + [windows[-1]]
        with self.assertRaisesRegex(ValueError, "missing, duplicate or noncontiguous"):
            build_flow_observation(trades, gap, decision_time=END + timedelta(seconds=2))

    def test_future_visibility_and_mixed_instruments_fail_closed(self):
        trades, windows = fixture()
        with self.assertRaisesRegex(ValueError, "not visible"):
            build_flow_observation(trades, windows, decision_time=END)
        with self.assertRaisesRegex(ValueError, "not visible"):
            build_flow_observation(
                [replace(trades[0], available_time=END + timedelta(seconds=3)), trades[1]],
                windows, decision_time=END + timedelta(seconds=2),
            )
        with self.assertRaisesRegex(ValueError, "mix"):
            build_flow_observation(
                [replace(trades[0], symbol="ETHUSDT"), trades[1]], windows,
                decision_time=END + timedelta(seconds=2),
            )
        with self.assertRaisesRegex(ValueError, "mix"):
            build_flow_observation(
                trades, windows[:-1] + [replace(windows[-1], venue="BYBIT")],
                decision_time=END + timedelta(seconds=2),
            )

    def test_trade_integrity_and_outcome_fields_rejected(self):
        trades, windows = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate trade"):
            build_flow_observation(
                [trades[0], replace(trades[1], event_id="buy-1")], windows,
                decision_time=END + timedelta(seconds=2),
            )
        with self.assertRaisesRegex(ValueError, "event-time order"):
            build_flow_observation(trades[::-1], windows, decision_time=END + timedelta(seconds=2))
        with self.assertRaisesRegex(ValueError, "outside"):
            build_flow_observation(
                [replace(trades[0], event_time=END + timedelta(seconds=1),
                         available_time=END + timedelta(seconds=1)), trades[1]],
                windows, decision_time=END + timedelta(seconds=2),
            )
        with self.assertRaisesRegex(ValueError, "quote amount"):
            build_flow_observation(
                [replace(trades[0], quantity=7), trades[1]], windows,
                decision_time=END + timedelta(seconds=2),
            )
        with self.assertRaisesRegex(ValueError, "trade count"):
            build_flow_observation(
                trades, windows[:-1] + [replace(windows[-1], trade_count=3)],
                decision_time=END + timedelta(seconds=2),
            )
        with self.assertRaises(TypeError):
            TradeEvent(**META, event_id="outcome", event_time=END,
                       available_time=END, price=1, quantity=1,
                       aggressor_side="BUY", post_alert_gain_pct=54.24)

    def test_optional_open_interest_requires_two_visible_bracketing_snapshots(self):
        trades, windows = fixture()
        start = OpenInterestSnapshot(
            **META, event_time=END - timedelta(minutes=5),
            available_time=END - timedelta(minutes=5) + timedelta(seconds=1),
            contracts=1000,
        )
        finish = OpenInterestSnapshot(
            **META, event_time=END, available_time=END + timedelta(seconds=2),
            contracts=1010,
        )
        result = build_flow_observation(
            trades, windows, decision_time=END + timedelta(seconds=3),
            open_interest_start=start, open_interest_end=finish,
        )
        self.assertAlmostEqual(result.observed_fields["open_interest_change_pct_5m"], 1.0)
        self.assertEqual(result.available_time, END + timedelta(seconds=2))
        with self.assertRaisesRegex(ValueError, "both start and end"):
            build_flow_observation(trades, windows, decision_time=END + timedelta(seconds=3),
                                   open_interest_start=start)
        with self.assertRaisesRegex(ValueError, "not visible"):
            build_flow_observation(
                trades, windows, decision_time=END + timedelta(seconds=3),
                open_interest_start=start,
                open_interest_end=replace(finish, available_time=END + timedelta(seconds=4)),
            )
        with self.assertRaisesRegex(ValueError, "bracket"):
            build_flow_observation(
                trades, windows, decision_time=END + timedelta(seconds=3),
                open_interest_start=start,
                open_interest_end=replace(finish, event_time=END - timedelta(seconds=1)),
            )

    def test_zero_trade_prior_window_is_attested_but_all_zero_baseline_rejected(self):
        trades, windows = fixture()
        windows[0] = replace(windows[0], trade_count=0, quote_volume_usd=0)
        result = build_flow_observation(trades, windows, decision_time=END + timedelta(seconds=2))
        self.assertAlmostEqual(
            result.observed_fields["quote_volume_baseline_5m_usd"], 1100 / 12,
        )
        windows = [replace(window, trade_count=0, quote_volume_usd=0)
                   for window in windows[:-1]] + [windows[-1]]
        with self.assertRaisesRegex(ValueError, "baseline is invalid"):
            build_flow_observation(trades, windows, decision_time=END + timedelta(seconds=2))


if __name__ == "__main__":
    unittest.main()
