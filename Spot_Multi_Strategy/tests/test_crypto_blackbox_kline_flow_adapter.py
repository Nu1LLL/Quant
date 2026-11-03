"""Synthetic-only tests for a checksum-verified 1m kline-to-label path."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest

from research.crypto_blackbox.binance_vision_5m import (
    ArchiveRequest, VerifiedKline, archive_url,
)
from research.crypto_blackbox.kline_flow_adapter import (
    KlineFlowAdapter, classify_verified_klines,
)


START = datetime(2025, 1, 31, 23, 0, tzinfo=timezone.utc)


def bar(index: int, *, start: datetime = START, symbol: str = "LSKUSDT",
        market: str = "spot", interval: str = "1m", quote: str = "20",
        buy: str = "10", count: int = 2) -> VerifiedKline:
    opened = start + timedelta(minutes=index)
    request = ArchiveRequest(market, symbol, opened.strftime("%Y-%m"), interval)
    url, _ = archive_url(request)
    digest = "a" * 64 if request.month == "2025-01" else "b" * 64
    duration = timedelta(minutes=1 if interval == "1m" else 5)
    return VerifiedKline(
        request=request, open_time=opened, window_end=opened + duration,
        quote_volume=Decimal(quote), taker_buy_quote_volume=Decimal(buy),
        trade_count=count, archive_sha256=digest, source_url=url,
    )


def alert_bars(count: int = 70):
    for index in range(count):
        if index >= 60:
            yield bar(index, quote="60", buy="36")
        else:
            yield bar(index)


class KlineFlowAdapterTests(unittest.TestCase):
    def test_cross_month_flow_math_and_first_then_continuation(self):
        rows = list(classify_verified_klines(
            alert_bars(), availability_lag=timedelta(seconds=2),
        ))
        # Twelve prior complete 5m windows, then two current windows.
        self.assertEqual(len(rows), 2)
        first, second = rows
        self.assertEqual(first.classification.label, "FIRST_ALERT")
        self.assertEqual(second.classification.label, "CONTINUATION")
        self.assertEqual(first.observation.market, "crypto_spot")
        self.assertEqual(first.observation.event_time,
                         datetime(2025, 2, 1, 0, 5, tzinfo=timezone.utc))
        self.assertEqual(first.observation.available_time,
                         first.observation.event_time + timedelta(seconds=2))
        self.assertEqual(first.classification.asof_time,
                         first.observation.available_time)
        self.assertEqual(first.observation.observed_fields["aggressor_buy_share_5m"], 0.6)
        self.assertEqual(first.observation.observed_fields["quote_volume_5m_usd"], 300.0)
        self.assertEqual(first.observation.observed_fields["quote_volume_baseline_5m_usd"],
                         100.0)
        self.assertEqual(first.observation.observed_fields["trade_count_5m"], 10)
        self.assertNotIn("open_interest_change_pct_5m", first.observation.observed_fields)
        self.assertIn("open_interest_unavailable", first.classification.reasons)
        self.assertEqual(tuple(digest for _, digest in first.archive_fingerprints),
                         ("a" * 64, "b" * 64))
        self.assertEqual(first.observation.source_version, second.observation.source_version)
        self.assertFalse(first.archive_publication_time_verified)
        self.assertTrue(first.research_only)
        self.assertFalse(first.deployable)
        self.assertEqual(first.assumed_availability_lag, timedelta(seconds=2))
        self.assertEqual(first.quote_asset, "USDT")

    def test_nonalert_gap_causes_next_spike_to_be_first(self):
        adapter = KlineFlowAdapter(availability_lag=timedelta(seconds=1))
        labels = []
        for index in range(135):
            spike = 60 <= index < 70 or 130 <= index < 135
            result = adapter.consume(bar(
                index, quote="60" if spike else "20",
                buy="36" if spike else "10",
            ))
            if result is not None:
                labels.append(result.classification.label)
        adapter.finish()
        self.assertEqual(labels[:2], ["FIRST_ALERT", "CONTINUATION"])
        self.assertEqual(labels[-1], "FIRST_ALERT")
        self.assertIn("NO_ALERT", labels[2:-1])

    def test_missing_or_duplicate_minute_halts_without_resynchronizing(self):
        for next_index in (6, 4):
            with self.subTest(next_index=next_index):
                adapter = KlineFlowAdapter(availability_lag=timedelta(seconds=1))
                for index in range(5):
                    adapter.consume(bar(index))
                with self.assertRaisesRegex(ValueError, "missing, duplicate"):
                    adapter.consume(bar(next_index))
                with self.assertRaisesRegex(RuntimeError, "halted"):
                    adapter.consume(bar(5))

    def test_stream_and_data_contracts_fail_closed(self):
        for bad_first in (
            bar(1),
            bar(0, interval="5m"),
            bar(0, symbol="ETHBTC"),
            bar(0, quote="0", buy="0", count=2),
        ):
            with self.subTest(bad_first=bad_first):
                with self.assertRaises(ValueError):
                    KlineFlowAdapter(availability_lag=timedelta(0)).consume(bad_first)
        adapter = KlineFlowAdapter(availability_lag=timedelta(0))
        adapter.consume(bar(0))
        with self.assertRaisesRegex(ValueError, "mixes market or symbol"):
            adapter.consume(bar(1, symbol="ETHUSDT"))
        with self.assertRaisesRegex(ValueError, "availability_lag"):
            KlineFlowAdapter(availability_lag=timedelta(minutes=5))
        with self.assertRaisesRegex(ValueError, "availability_lag"):
            KlineFlowAdapter(availability_lag=timedelta(seconds=-1))
        revised = KlineFlowAdapter(availability_lag=timedelta(seconds=1))
        revised.consume(bar(0))
        with self.assertRaisesRegex(ValueError, "conflicting SHA-256"):
            revised.consume(replace(bar(1), archive_sha256="c" * 64))

    def test_zero_volume_window_is_not_an_alert_but_remains_in_baseline(self):
        adapter = KlineFlowAdapter(availability_lag=timedelta(seconds=1))
        rows = []
        for index in range(70):
            if 60 <= index < 65:
                row = bar(index, quote="0", buy="0", count=0)
            elif index >= 65:
                row = bar(index, quote="60", buy="36")
            else:
                row = bar(index)
            outcome = adapter.consume(row)
            if outcome is not None:
                rows.append(outcome)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].classification.label, "FIRST_ALERT")
        self.assertAlmostEqual(
            rows[0].observation.observed_fields["quote_volume_baseline_5m_usd"],
            1100 / 12,
        )

    def test_truncated_five_minute_window_and_perp_metadata(self):
        adapter = KlineFlowAdapter(availability_lag=timedelta(seconds=1))
        adapter.consume(bar(0))
        with self.assertRaisesRegex(ValueError, "incomplete five-minute"):
            adapter.finish()
        rows = list(classify_verified_klines(
            (bar(i, market="um") if i < 60 else
             bar(i, market="um", quote="60", buy="36") for i in range(65)),
            availability_lag=timedelta(seconds=1),
        ))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].observation.market, "crypto_perpetual")
        self.assertNotIn("open_interest_change_pct_5m", rows[0].observation.observed_fields)


if __name__ == "__main__":
    unittest.main()
