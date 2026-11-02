import unittest

import numpy as np
import pandas as pd

import microstructure_fill_simulator as fills
from tests.test_microstructure_blackbox import MicrostructureBlackBoxTests


class MicrostructureFillSimulatorTests(unittest.TestCase):
    @staticmethod
    def signals(frame, actions, cost=3.0):
        index = frame.index[:len(actions)]
        return pd.DataFrame({
            "market": "crypto", "venue": "X", "symbol": "BTC",
            "alpha_id": "a", "action": actions, "raw_action": actions,
            "cost_gate_bps": cost, "valid_for_seconds": 60,
            "kelly_position": 0.1, "model_version": "v1",
        }, index=index)

    @staticmethod
    def flat_mid_frame(mids):
        frame = MicrostructureBlackBoxTests.frame(len(mids))
        frame.index = pd.date_range("2026-01-01", periods=len(mids), freq="min", tz="UTC")
        mids = np.asarray(mids, dtype=float)
        frame["best_bid"] = mids - 0.005
        frame["best_ask"] = mids + 0.005
        return frame

    def test_flat_market_loses_for_buy_and_sell(self):
        frame = self.flat_mid_frame([100, 100, 100])
        result = fills.simulate_taker_fills(self.signals(frame, ["BUY", "SELL"]), frame)
        self.assertTrue((result["net_return"] < 0).all())

    def test_rising_market_favors_buy_and_higher_cost_lowers_return(self):
        frame = self.flat_mid_frame([100, 101, 102])
        signals = self.signals(frame, ["BUY", "SELL"], cost=3)
        low = fills.simulate_taker_fills(signals, frame)
        high_signals = signals.copy()
        high_signals["cost_gate_bps"] = 30
        high = fills.simulate_taker_fills(high_signals, frame)
        self.assertGreater(low.iloc[0]["net_return"], low.iloc[1]["net_return"])
        self.assertTrue((high["net_return"] < low["net_return"]).all())

    def test_flat_and_expired_signals_do_not_fill(self):
        frame = self.flat_mid_frame([100, 100, 100])
        signals = self.signals(frame, ["FLAT", "BUY"])
        signals["valid_for_seconds"] = 59
        self.assertTrue(fills.simulate_taker_fills(signals, frame).empty)

    def test_future_change_does_not_rewrite_prior_fill(self):
        frame = self.flat_mid_frame([100, 101, 102, 103])
        signals = self.signals(frame, ["BUY"])
        original = fills.simulate_taker_fills(signals, frame)
        changed = frame.copy()
        changed.iloc[2:, changed.columns.get_loc("best_bid")] *= 10
        changed.iloc[2:, changed.columns.get_loc("best_ask")] *= 10
        revised = fills.simulate_taker_fills(signals, changed)
        pd.testing.assert_frame_equal(original, revised)


if __name__ == "__main__":
    unittest.main()
