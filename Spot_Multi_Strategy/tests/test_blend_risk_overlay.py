import unittest

import numpy as np
import pandas as pd

import blend_risk_overlay


class BlendRiskOverlayTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(99)
        index = pd.date_range("2016-09-12", periods=1200, freq="B", tz="UTC")
        self.returns = pd.Series(
            rng.normal(0.0005, 0.01, len(index)), index=index
        )

    def test_future_mutation_does_not_change_past(self):
        before, _, _ = blend_risk_overlay.run_overlay(self.returns)
        changed = self.returns.copy()
        changed.iloc[900:] *= 20.0
        after, _, _ = blend_risk_overlay.run_overlay(changed)
        columns = ["raw_vol_target", "drawdown_scalar", "position", "net_pnl"]
        pd.testing.assert_frame_equal(
            before.iloc[:900][columns], after.iloc[:900][columns]
        )

    def test_position_respects_cap(self):
        simulation, _, _ = blend_risk_overlay.run_overlay(
            self.returns, vol_target=10.0, exposure_cap=3.0
        )
        self.assertLessEqual(float(simulation["position"].max()), 3.0)

    def test_drawdown_scalar_boundaries(self):
        self.assertEqual(blend_risk_overlay.drawdown_scalar(-0.08), 1.0)
        self.assertEqual(blend_risk_overlay.drawdown_scalar(-0.20), 0.30)
        self.assertGreater(
            blend_risk_overlay.drawdown_scalar(-0.10),
            blend_risk_overlay.drawdown_scalar(-0.15),
        )

    def test_costs_are_nonnegative(self):
        simulation, _, metrics = blend_risk_overlay.run_overlay(self.returns)
        self.assertTrue((simulation["cost"] >= 0).all())
        self.assertIn("worst_rolling_3y_sharpe", metrics)


if __name__ == "__main__":
    unittest.main()
