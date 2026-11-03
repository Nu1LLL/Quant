"""Synthetic proof-of-contract tests; no historical market values are loaded."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest

from research.crypto_blackbox.regime_context import (
    BTC_MVRV,
    BTC_MA_50W_USD,
    BTC_MA_200W_USD,
    BTC_SPOT_USD,
    METRIC_UNITS,
    US10Y_YIELD_PCT,
    US_M2_USD_BILLIONS,
    ContextPoint,
    RegimeContextConfig,
    VintageEvidence,
    build_regime_context,
)


ASOF = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)
VALUES = {
    BTC_SPOT_USD: 100.0,
    BTC_MVRV: 1.2,
    BTC_MA_50W_USD: 90.0,
    BTC_MA_200W_USD: 80.0,
    US10Y_YIELD_PCT: 4.0,
    US_M2_USD_BILLIONS: 20_000.0,
}
AGES = {
    BTC_SPOT_USD: timedelta(minutes=5),
    BTC_MVRV: timedelta(days=1),
    BTC_MA_50W_USD: timedelta(days=3),
    BTC_MA_200W_USD: timedelta(days=3),
    US10Y_YIELD_PCT: timedelta(days=2),
    US_M2_USD_BILLIONS: timedelta(days=30),
}


def synthetic_point(metric, *, age=None, value=None):
    available = ASOF - (AGES[metric] if age is None else age)
    vintage = VintageEvidence(
        vintage_id=f"synthetic-vintage-{metric}-{available.date().isoformat()}",
        published_at=available - timedelta(minutes=2),
        first_archived_at=available - timedelta(minutes=1),
        archive_uri=f"https://archive.example.test/{metric}/{available.date().isoformat()}",
        sha256="a" * 64,
    )
    return ContextPoint(
        metric=metric, value=VALUES[metric] if value is None else value,
        unit=METRIC_UNITS[metric],
        observed_at=available - timedelta(minutes=3), available_at=available,
        source="synthetic-primary", source_version="v1", license_id="test-license",
        vintage=vintage,
    )


def six_points():
    return [synthetic_point(metric) for metric in VALUES]


class RegimeContextTests(unittest.TestCase):
    def build(self, points=None, **kwargs):
        return build_regime_context(
            six_points() if points is None else points,
            ASOF, "crypto_perpetual", "BTCUSDT", **kwargs,
        )

    def test_only_arithmetic_labels_and_auditable_raw_points(self):
        result = self.build()
        self.assertEqual(result.labels, (
            "btc_mvrv_above_one", "btc_spot_vs_50w_ma_above",
            "btc_spot_vs_200w_ma_above", "btc_50w_ma_vs_200w_ma_above",
        ))
        self.assertEqual(len(result.points), 6)
        self.assertEqual({p.metric for p in result.points}, set(VALUES))
        self.assertTrue(all(p.source and p.source_version and p.license_id
                            and p.vintage.sha256 and p.available_at <= ASOF
                            for p in result.points))
        self.assertTrue(result.read_only)
        self.assertFalse(result.tradable)
        for forbidden in ("action", "target_weight", "expected_return", "pnl", "forecast"):
            self.assertFalse(hasattr(result, forbidden))
        reversed_result = self.build(reversed(six_points()))
        self.assertEqual(result.points, reversed_result.points)
        self.assertEqual(result.labels, reversed_result.labels)

    def test_neutral_and_below_labels_are_descriptive_not_bottom_calls(self):
        points = six_points()
        changed = []
        for point in points:
            if point.metric == BTC_MVRV:
                point = replace(point, value=1.0)
            if point.metric == BTC_SPOT_USD:
                point = replace(point, value=70.0)
            changed.append(point)
        labels = self.build(changed).labels
        self.assertEqual(labels[0], "btc_mvrv_equal_one")
        self.assertEqual(labels[1], "btc_spot_vs_50w_ma_below")
        self.assertNotIn("BUY", labels)
        self.assertNotIn("BOTTOM", labels)

    def test_missing_duplicate_and_noncrypto_targets_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.build(six_points()[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate context"):
            self.build(six_points() + [synthetic_point(BTC_MVRV)])
        with self.assertRaisesRegex(ValueError, "crypto trading target"):
            build_regime_context(six_points(), ASOF, "equity", "SPY")
        with self.assertRaisesRegex(ValueError, "canonical crypto symbol"):
            build_regime_context(six_points(), ASOF, "crypto_spot", "EUR/USD")

    def test_future_availability_or_retrospective_archive_fails_closed(self):
        points = six_points()
        future = synthetic_point(BTC_SPOT_USD, age=-timedelta(minutes=2))
        points[0] = future
        with self.assertRaisesRegex(ValueError, "not archived at decision"):
            self.build(points)
        with self.assertRaisesRegex(ValueError, "available at decision"):
            self.build([replace(synthetic_point(BTC_SPOT_USD),
                                available_at=ASOF + timedelta(minutes=1))]
                       + six_points()[1:])
        with self.assertRaisesRegex(ValueError, "UTC"):
            build_regime_context(six_points(), ASOF.replace(tzinfo=None),
                                 "crypto_spot", "BTCUSDT")

    def test_stale_macro_and_technical_points_fail_closed(self):
        for metric, stale_age in (
            (BTC_SPOT_USD, timedelta(hours=2)),
            (BTC_MVRV, timedelta(days=4)),
            (BTC_MA_50W_USD, timedelta(days=11)),
            (US10Y_YIELD_PCT, timedelta(days=8)),
            (US_M2_USD_BILLIONS, timedelta(days=76)),
        ):
            with self.subTest(metric=metric):
                points = [synthetic_point(name, age=stale_age if name == metric else None)
                          for name in VALUES]
                with self.assertRaisesRegex(ValueError, "stale"):
                    self.build(points)

    def test_missing_or_fake_vintage_and_provenance_fail_closed(self):
        point = synthetic_point(US_M2_USD_BILLIONS)
        with self.assertRaisesRegex(ValueError, "vintage evidence"):
            replace(point, vintage=None)
        with self.assertRaisesRegex(ValueError, "historical vintage"):
            replace(point.vintage, vintage_id="latest")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            replace(point.vintage, sha256="not-a-digest")
        with self.assertRaisesRegex(ValueError, "archive_uri"):
            replace(point.vintage, archive_uri="")
        with self.assertRaisesRegex(ValueError, "license_id"):
            replace(point, license_id="")
        with self.assertRaisesRegex(ValueError, "source_version"):
            replace(point, source_version="")
        with self.assertRaisesRegex(ValueError, "publication"):
            replace(point.vintage,
                    published_at=point.vintage.first_archived_at + timedelta(minutes=1))

    def test_unit_value_and_config_contract(self):
        point = synthetic_point(BTC_MVRV)
        with self.assertRaisesRegex(ValueError, "unit mismatch"):
            replace(point, unit="percent")
        with self.assertRaisesRegex(ValueError, "positive"):
            replace(point, value=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(point, value=float("nan"))
        with self.assertRaisesRegex(ValueError, "positive timedelta"):
            RegimeContextConfig(m2_max_age=timedelta(0))
        with self.assertRaisesRegex(ValueError, "order is invalid"):
            replace(point, observed_at=point.available_at + timedelta(seconds=1))


if __name__ == "__main__":
    unittest.main()
