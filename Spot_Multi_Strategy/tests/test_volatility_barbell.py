import unittest

import numpy as np
import pandas as pd

import fixed_weight_blend
import volatility_barbell_report


class VolatilityBarbellTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(
            volatility_barbell_report.START, volatility_barbell_report.END, freq="B"
        )
        n = len(self.index)
        self.growth = pd.Series(0.0005 + 0.001*np.sin(np.arange(n)), index=self.index)
        self.svrpo = pd.Series(100*np.exp(np.linspace(0, 0.5, n)), index=self.index)
        self.vxth = pd.Series(100*np.exp(np.linspace(0, 0.2, n)), index=self.index)

    def test_target_weights_and_schedule(self):
        levels = volatility_barbell_report.build_levels(self.growth, self.svrpo, self.vxth)
        _, positions, turnover = fixed_weight_blend.run_fixed_weight(
            levels, {"GROWTH": 0.5, "SVRPO": 0.25, "VXTH": 0.25}
        )
        self.assertTrue(np.allclose(positions.iloc[::21], [0.5, 0.25, 0.25]))
        self.assertAlmostEqual(turnover.iloc[0], 1.0)
        self.assertTrue((turnover >= 0).all())

    def test_future_mutation_does_not_change_past(self):
        before = volatility_barbell_report.build_levels(self.growth, self.svrpo, self.vxth)
        changed = self.vxth.copy(); changed.iloc[1500:] *= 3
        after = volatility_barbell_report.build_levels(self.growth, self.svrpo, changed)
        pd.testing.assert_frame_equal(before.iloc[:1500], after.iloc[:1500])

    def test_six_scenarios_are_fixed(self):
        summary, artifacts, _ = volatility_barbell_report.run_experiment(
            self.growth, self.svrpo, self.vxth, repetitions=39
        )
        expected = {"VOL_BARBELL_1x", "VOL_BARBELL_RISK_OVERLAY",
                    "FULL_1x", "FULL_2x", "FULL_3x", "FULL_RISK_OVERLAY"}
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertIn("passed", summary)

    def test_invalid_weights_rejected(self):
        levels = volatility_barbell_report.build_levels(self.growth, self.svrpo, self.vxth)
        with self.assertRaises(ValueError):
            fixed_weight_blend.run_fixed_weight(levels, {"GROWTH": .7, "SVRPO": .2, "VXTH": .2})


if __name__ == "__main__":
    unittest.main()
