import unittest

import numpy as np
import pandas as pd

import microstructure_signal_runner as runner
from tests.test_microstructure_blackbox import MicrostructureBlackBoxTests


class MicrostructureSignalRunnerTests(unittest.TestCase):
    def test_runner_returns_market_qualified_signals(self):
        frame = MicrostructureBlackBoxTests.frame(1800)
        frame["market"] = "crypto"
        frame["venue"] = "synthetic"
        frame["symbol"] = "BTCUSDT"
        frame["fee_bps_each_side"] = 2.0
        frame["impact_bps_each_side"] = 0.5
        frame["financing_bps_horizon"] = 0.0
        result = runner.run_symbol_frame(frame, frame.index[1199], "test-v2")
        self.assertTrue((result["market"] == "crypto").all())
        self.assertTrue((result["venue"] == "synthetic").all())
        self.assertTrue((result["symbol"] == "BTCUSDT").all())
        self.assertTrue((result["alpha_id"] == "hmm-rbf-minute-v2").all())
        self.assertTrue(result["model_version"].str.startswith("test-v2:").all())
        self.assertTrue(np.isfinite(result["kelly_position"]).all())
        self.assertFalse(result["deployable"].any())

    def test_lifecycle_gate_flattens_unvalidated_and_sizes_active(self):
        index = pd.date_range("2026-01-01", periods=2, freq="min", tz="UTC")
        orders = pd.DataFrame({
            "market": ["fx", "equity"], "venue": ["D", "N"],
            "symbol": ["EURUSD", "SPY"], "alpha_id": ["a", "a"],
            "action": ["BUY", "SELL"], "raw_action": ["BUY", "SELL"],
            "kelly_position": [0.20, -0.20],
        }, index=index)
        table = pd.DataFrame({
            "market": ["fx", "equity"], "venue": ["D", "N"],
            "symbol": ["EURUSD", "SPY"], "alpha_id": ["a", "a"],
            "status": ["ACTIVE", "DISABLED"], "weight": [0.10, 0.0],
        })
        gated = runner.gate_orders_with_lifecycle(orders, table)
        self.assertEqual(gated.iloc[0]["action"], "BUY")
        self.assertAlmostEqual(gated.iloc[0]["kelly_position"], 0.02)
        self.assertEqual(gated.iloc[1]["action"], "FLAT")
        self.assertEqual(gated.iloc[1]["kelly_position"], 0.0)
