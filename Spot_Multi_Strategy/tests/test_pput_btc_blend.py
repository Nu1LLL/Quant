import unittest

import numpy as np
import pandas as pd

import pput_btc_blend as blend


class PputBtcBlendTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(23)
        self.index = pd.date_range("2016-09-12", periods=1000, freq="B", tz="UTC")
        self.returns = pd.DataFrame({
            "PPUT": rng.normal(0.0003, 0.008, len(self.index)),
            "BTC_TSMOM30_DD": rng.normal(0.0008, 0.025, len(self.index)),
        }, index=self.index)

    def test_positions_are_causal(self):
        before = blend.build_positions(self.returns)
        changed = self.returns.copy()
        changed.iloc[700:] *= 10.0
        after = blend.build_positions(changed)
        pd.testing.assert_frame_equal(before.iloc[:700], after.iloc[:700])

    def test_active_weights_sum_to_one(self):
        positions = blend.build_positions(self.returns)
        active = positions.sum(axis=1) > 0
        np.testing.assert_allclose(
            positions.loc[active].sum(axis=1).to_numpy(), 1.0
        )

    def test_weekend_equity_change_is_not_dropped(self):
        cboe_dates = pd.to_datetime(
            ["2024-01-05", "2024-01-08"], utc=True
        )
        pput = pd.Series([100.0, 100.0], index=cboe_dates)
        btc_dates = pd.date_range("2024-01-05", "2024-01-08", freq="D", tz="UTC")
        btc = pd.DataFrame({
            "open_time": btc_dates,
            "equity": [1.0, 1.1, 1.21, 1.331],
        })
        returns = blend.align_component_returns(pput, btc)
        self.assertAlmostEqual(returns["BTC_TSMOM30_DD"].iloc[1], 0.331)

    def test_scenario_financing_nonnegative(self):
        simulation, positions, yearly, metrics = blend.run_scenario(
            self.returns, leverage=2.0
        )
        self.assertTrue((simulation["financing_cost"] >= 0).all())
        self.assertIn("sharpe_ratio", metrics)


if __name__ == "__main__":
    unittest.main()
