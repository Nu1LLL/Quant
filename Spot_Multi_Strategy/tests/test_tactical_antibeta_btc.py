import unittest

import numpy as np
import pandas as pd

import tactical_antibeta_btc
import tactical_antibeta_btc_report


class TacticalAntibetaBtcTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(
            "2016-09-12", periods=1200, freq="B", tz="UTC"
        )
        self.spy = pd.Series(
            np.concatenate([
                np.linspace(100.0, 180.0, 600),
                np.linspace(180.0, 80.0, 600),
            ]), index=self.index
        )
        self.btal = pd.Series(
            np.linspace(100.0, 115.0, len(self.index)), index=self.index
        )

    def test_switches_only_after_month_end(self):
        _, positions, _ = tactical_antibeta_btc.build_tactical_sleeve(
            self.spy, self.btal
        )
        changes = positions.diff().abs().sum(axis=1).gt(0.0)
        for date in positions.index[changes]:
            previous = positions.index[positions.index.get_loc(date) - 1]
            self.assertNotEqual(previous.month, date.month)

    def test_future_mutation_does_not_change_past(self):
        before = tactical_antibeta_btc.build_tactical_sleeve(
            self.spy, self.btal
        )
        spy = self.spy.copy()
        btal = self.btal.copy()
        spy.iloc[900:] *= 3.0
        btal.iloc[900:] *= 0.5
        after = tactical_antibeta_btc.build_tactical_sleeve(spy, btal)
        for before_item, after_item in zip(before, after):
            if isinstance(before_item, pd.DataFrame):
                pd.testing.assert_frame_equal(
                    before_item.iloc[:900], after_item.iloc[:900]
                )
            else:
                pd.testing.assert_series_equal(
                    before_item.iloc[:900], after_item.iloc[:900]
                )

    def test_positions_are_cash_or_fully_in_one_etf(self):
        _, positions, _ = tactical_antibeta_btc.build_tactical_sleeve(
            self.spy, self.btal
        )
        self.assertTrue((positions >= 0.0).all().all())
        self.assertTrue(positions.sum(axis=1).isin([0.0, 1.0]).all())
        self.assertTrue(((positions > 0.0).sum(axis=1) <= 1).all())

    def test_experiment_has_fixed_scenarios(self):
        rng = np.random.default_rng(1013)
        returns = pd.DataFrame({
            "TACTICAL_ANTIBETA": rng.normal(0.0003, 0.009, 1000),
            "BTC_TSMOM30_DD": rng.normal(0.0008, 0.024, 1000),
        }, index=self.index[:1000])
        summary, artifacts = tactical_antibeta_btc_report.run_experiment(
            returns, repetitions=99
        )
        expected = {"1x", "2x", "3x", "RISK_OVERLAY"}
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
