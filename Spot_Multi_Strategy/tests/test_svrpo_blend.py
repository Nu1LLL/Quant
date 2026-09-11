import unittest

import numpy as np
import pandas as pd

import svrpo_blend
import svrpo_blend_report


class SvrpoBlendTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(svrpo_blend_report.START, svrpo_blend_report.END, freq="B")
        self.growth = pd.Series(0.0005 + 0.001 * np.sin(np.arange(len(self.index))), index=self.index)
        self.svrpo = pd.Series(100 * np.exp(np.linspace(0, 1, len(self.index))), index=self.index)

    def test_full_window_is_required(self):
        with self.assertRaises(ValueError):
            svrpo_blend.align_levels(
                self.growth, self.svrpo.iloc[:-1],
                svrpo_blend_report.START, svrpo_blend_report.END
            )

    def test_equal_capital_rebalances_on_fixed_schedule(self):
        levels = svrpo_blend.align_levels(
            self.growth, self.svrpo, svrpo_blend_report.START, svrpo_blend_report.END
        )
        _, positions, turnover = svrpo_blend.equal_capital_returns(levels)
        self.assertAlmostEqual(turnover.iloc[0], 1.0)
        self.assertTrue(np.allclose(positions.iloc[::21].values, 0.5))
        self.assertTrue((turnover >= 0).all())

    def test_future_mutation_does_not_change_past(self):
        before = svrpo_blend.align_levels(
            self.growth, self.svrpo, svrpo_blend_report.START, svrpo_blend_report.END
        )
        changed = self.svrpo.copy(); changed.iloc[1500:] *= 2
        after = svrpo_blend.align_levels(
            self.growth, changed, svrpo_blend_report.START, svrpo_blend_report.END
        )
        pd.testing.assert_frame_equal(before.iloc[:1500], after.iloc[:1500])

    def test_seven_scenarios_are_fixed(self):
        summary, artifacts, _ = svrpo_blend_report.run_experiment(
            self.growth, self.svrpo, repetitions=39
        )
        expected = {"SVRPO_1x", "SVRPO_2x", "SVRPO_3x", "BLEND_1x",
                    "BLEND_2x", "BLEND_3x", "BLEND_RISK_OVERLAY"}
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
