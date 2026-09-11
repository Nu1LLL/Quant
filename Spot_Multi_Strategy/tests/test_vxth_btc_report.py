import unittest

import numpy as np
import pandas as pd

import vxth_btc_report


class VxthBtcReportTests(unittest.TestCase):
    def test_experiment_has_fixed_scenarios_and_gate_columns(self):
        rng = np.random.default_rng(887)
        index = pd.date_range("2016-09-12", periods=1000, freq="B", tz="UTC")
        returns = pd.DataFrame({
            "VXTH": rng.normal(0.0003, 0.009, len(index)),
            "BTC_TSMOM30_DD": rng.normal(0.0008, 0.024, len(index)),
        }, index=index)
        summary, artifacts = vxth_btc_report.run_experiment(
            returns, repetitions=99
        )
        self.assertEqual(
            set(summary["scenario"]), {"1x", "2x", "3x", "RISK_OVERLAY"}
        )
        self.assertEqual(set(artifacts), set(summary["scenario"]))
        self.assertIn("check__worst_rolling_3y_sharpe_1", summary)
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
