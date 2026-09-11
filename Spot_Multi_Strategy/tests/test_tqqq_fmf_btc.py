import unittest

import numpy as np
import pandas as pd

import tqqq_fmf_btc
import tqqq_fmf_btc_report


class TqqqFmfBtcTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2015-09-01", periods=2900, freq="B", tz="UTC")
        self.qqq = pd.Series(100 * np.exp(np.linspace(0, .8, 2900)), index=self.index)
        self.tqqq = pd.Series(100 * np.exp(np.linspace(0, 2, 2900)), index=self.index)
        self.fmf = pd.Series(100 * np.exp(np.linspace(0, .3, 2900)), index=self.index)
        dates = pd.date_range("2016-09-12", "2026-09-10", freq="D", tz="UTC")
        self.btc = pd.DataFrame({"open_time": dates, "equity": np.exp(np.linspace(0, 1, len(dates)))})

    def test_positions_are_causal_and_exclusive(self):
        before = tqqq_fmf_btc.build_switch_positions(self.qqq, self.tqqq, self.fmf)
        changed = self.qqq.copy()
        changed.iloc[1800:] *= 3.0
        after = tqqq_fmf_btc.build_switch_positions(changed, self.tqqq, self.fmf)
        pd.testing.assert_frame_equal(before.iloc[:1800], after.iloc[:1800])
        self.assertTrue(before.sum(axis=1).isin([0.0, 1.0]).all())
        self.assertTrue(((before > 0).sum(axis=1) <= 1).all())

    def test_changes_apply_after_month_end(self):
        positions = tqqq_fmf_btc.build_switch_positions(self.qqq, self.tqqq, self.fmf)
        for date in positions.index[positions.diff().abs().sum(axis=1).gt(0)]:
            previous = positions.index[positions.index.get_loc(date) - 1]
            self.assertNotEqual(previous.month, date.month)

    def test_experiment_has_six_scenarios(self):
        summary, artifacts, components = tqqq_fmf_btc_report.run_experiment(
            self.qqq, self.tqqq, self.fmf, self.btc, repetitions=99
        )
        expected = {
            "TQQQ_FMF_SWITCH", "SWITCH_RISK_OVERLAY", "BLEND_1x",
            "BLEND_2x", "BLEND_3x", "BLEND_RISK_OVERLAY",
        }
        self.assertEqual(set(summary.scenario), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertEqual(list(components), ["TQQQ_FMF_SWITCH", "BTC_TSMOM30_DD"])


if __name__ == "__main__":
    unittest.main()
