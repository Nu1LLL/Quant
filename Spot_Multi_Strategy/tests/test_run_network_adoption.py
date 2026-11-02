from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from research.network_adoption_binance import Bar, Funding, MonthArchive

from research.run_network_adoption import (
    NOTICES, SYMBOLS, StudyDataError, _json_safe, _months, archive_jobs,
    load_archives, next_4h_open_after,
)


UTC = timezone.utc


class NetworkStudyRunnerTests(unittest.TestCase):
    def test_strictly_later_exit_bar(self):
        self.assertEqual(
            next_4h_open_after(datetime(2024, 12, 6, 3, 0, tzinfo=UTC)),
            datetime(2024, 12, 6, 4, 0, tzinfo=UTC),
        )
        self.assertEqual(
            next_4h_open_after(datetime(2024, 12, 6, 4, 0, tzinfo=UTC)),
            datetime(2024, 12, 6, 8, 0, tzinfo=UTC),
        )

    def test_months_across_year_end(self):
        self.assertEqual(
            _months(datetime(2024, 11, 1, tzinfo=UTC),
                    datetime(2025, 1, 1, tzinfo=UTC)),
            ("2024-11", "2024-12", "2025-01"),
        )

    def test_archive_jobs_keep_delisted_cohort_and_no_relisted_cvc(self):
        jobs = archive_jobs()
        self.assertEqual(set(symbol for _, symbol, _ in jobs if symbol != "BTCUSDT"),
                         set(SYMBOLS))
        for symbol in NOTICES:
            last_month = next_4h_open_after(NOTICES[symbol]).strftime("%Y-%m")
            self.assertIn(("trade4", symbol, last_month), jobs)
        self.assertNotIn(("trade4", "CVCUSDT", "2025-05"), jobs)
        self.assertIn(("trade4", "BATUSDT", "2026-01"), jobs)
        self.assertIn(("btc_spot1d", "BTCUSDT", "2020-12"), jobs)
        self.assertIn(("btc_spot1d", "BTCUSDT", "2025-12"), jobs)
        self.assertNotIn(("btc_spot1d", "BTCUSDT", "2026-01"), jobs)

    def test_json_safe_infinity_not_nonstandard_nan(self):
        self.assertEqual(_json_safe(float("inf")), "Infinity")
        with self.assertRaisesRegex(StudyDataError, "non-finite result"):
            _json_safe(float("nan"))

    def test_missing_funding_across_month_boundary_is_rejected(self):
        # Exercise the runner's continuity gate using only invented bars and
        # funding rows.  The test does not request any real test-period data.
        first = datetime(2020, 12, 1, tzinfo=UTC)
        terminal = datetime(2021, 1, 2, tzinfo=UTC)
        symbol = "BATUSDT"
        four_hour_count = int((terminal - first) / timedelta(hours=4)) + 1
        one_hour_count = int((terminal - first) / timedelta(hours=1)) + 1
        funding_count = int((terminal - first) / timedelta(hours=8)) + 1

        def bars(period, count, *, quote_volume):
            return tuple(
                Bar(first + period * index, 100.0, 100.0, 100.0, 100.0,
                    quote_volume)
                for index in range(count)
            )

        full_funding = tuple(
            Funding(first + timedelta(hours=8 * index), 0.0001, 8.0)
            for index in range(funding_count)
        )
        btc_start = datetime(2020, 12, 1, tzinfo=UTC)
        btc_bars = tuple(
            Bar(btc_start + timedelta(days=index), 100.0, 100.0,
                100.0, 100.0, 1_000_000.0)
            for index in range(1857)
        )
        jobs = (
            ("trade4", symbol, "2020-12"),
            ("mark4", symbol, "2020-12"),
            ("mark1", symbol, "2020-12"),
            ("funding", symbol, "2020-12"),
            ("funding", symbol, "2021-01"),
            ("btc_spot1d", "BTCUSDT", "2020-12"),
        )

        def run_with_funding(events):
            by_month = {
                "2020-12": tuple(item for item in events if item.calc_time.month == 12),
                "2021-01": tuple(item for item in events if item.calc_time.month == 1),
            }
            rows = {
                jobs[0]: bars(timedelta(hours=4), four_hour_count, quote_volume=1_000_000.0),
                jobs[1]: bars(timedelta(hours=4), four_hour_count, quote_volume=None),
                jobs[2]: bars(timedelta(hours=1), one_hour_count, quote_volume=None),
                jobs[3]: by_month["2020-12"],
                jobs[4]: by_month["2021-01"],
                jobs[5]: btc_bars,
            }

            def fake_fetch(job):
                return job, MonthArchive(rows[job], "a" * 64, 100, "synthetic://archive")

            with patch("research.run_network_adoption.SYMBOLS", (symbol,)), \
                 patch("research.run_network_adoption.TERMINAL", terminal), \
                 patch("research.run_network_adoption.archive_jobs", return_value=jobs), \
                 patch("research.run_network_adoption._fetch_archive_job", side_effect=fake_fetch):
                return load_archives(max_workers=1)

        self.assertEqual(len(run_with_funding(full_funding).funding[symbol]), funding_count)
        omitted_january_open = tuple(
            item for item in full_funding
            if item.calc_time != datetime(2021, 1, 1, tzinfo=UTC)
        )
        with self.assertRaisesRegex(StudyDataError, "missing/ambiguous funding event"):
            run_with_funding(omitted_january_open)


if __name__ == "__main__":
    unittest.main()
