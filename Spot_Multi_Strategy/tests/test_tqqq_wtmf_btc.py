import unittest

import numpy as np
import pandas as pd

import tqqq_wtmf_btc
import tqqq_wtmf_btc_report


class TqqqWtmfBtcTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(
            "2015-09-01", periods=2900, freq="B", tz="UTC"
        )
        self.qqq = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 0.8, len(self.index))),
            index=self.index,
        )
        self.tqqq = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 2.0, len(self.index))),
            index=self.index,
        )
        self.wtmf = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 0.3, len(self.index))),
            index=self.index,
        )
        btc_dates = pd.date_range(
            "2016-09-12", "2026-09-10", freq="D", tz="UTC"
        )
        self.btc = pd.DataFrame({
            "open_time": btc_dates,
            "equity": np.exp(np.linspace(0.0, 1.0, len(btc_dates))),
        })

    def test_switches_only_after_month_end(self):
        positions = tqqq_wtmf_btc.build_switch_positions(
            self.qqq, self.tqqq, self.wtmf
        )
        changes = positions.diff().abs().sum(axis=1).gt(0.0)
        for date in positions.index[changes]:
            previous = positions.index[positions.index.get_loc(date) - 1]
            self.assertNotEqual(previous.month, date.month)

    def test_future_mutation_does_not_change_past(self):
        before = tqqq_wtmf_btc.build_switch_positions(
            self.qqq, self.tqqq, self.wtmf
        )
        qqq = self.qqq.copy()
        wtmf = self.wtmf.copy()
        qqq.iloc[1800:] *= 3.0
        wtmf.iloc[1800:] *= 0.5
        after = tqqq_wtmf_btc.build_switch_positions(
            qqq, self.tqqq, wtmf
        )
        pd.testing.assert_frame_equal(before.iloc[:1800], after.iloc[:1800])

    def test_positions_are_cash_or_one_etf(self):
        positions = tqqq_wtmf_btc.build_switch_positions(
            self.qqq, self.tqqq, self.wtmf
        )
        self.assertTrue(positions.sum(axis=1).isin([0.0, 1.0]).all())
        self.assertTrue(((positions > 0.0).sum(axis=1) <= 1).all())

    def test_experiment_has_six_fixed_scenarios(self):
        summary, artifacts, components = tqqq_wtmf_btc_report.run_experiment(
            self.qqq, self.tqqq, self.wtmf, self.btc, repetitions=99
        )
        expected = {
            "TQQQ_WTMF_SWITCH", "SWITCH_RISK_OVERLAY",
            "BLEND_1x", "BLEND_2x", "BLEND_3x", "BLEND_RISK_OVERLAY",
        }
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertEqual(
            list(components), ["TQQQ_WTMF_SWITCH", "BTC_TSMOM30_DD"]
        )
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
