import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))


from paper_trader import PaperDataGapError
from rotation_paper_trader import (
    create_rotation_paper_state,
    process_rotation_state
)
from rotation_strategy import RotationConfig


def processed_frames(signal_time):
    return {
        "BTCUSDT": pd.DataFrame(
            {
                "open_time": [signal_time],
                "eligible": [True],
                "realized_volatility": [0.48]
            }
        ),
        "ETHUSDT": pd.DataFrame(
            {
                "open_time": [signal_time],
                "eligible": [True],
                "realized_volatility": [0.30]
            }
        )
    }


def candidate_config():
    return RotationConfig(
        slow_ema_window=8,
        fast_ema_window=4,
        momentum_window=5,
        ema_slope_window=2,
        rebalance_bars=1,
        exposure=1.0,
        broad_market_exposure=1.0,
        volatility_window=10,
        target_volatility=0.24,
        minimum_exposure=0.10
    )


class RotationPaperTraderTests(unittest.TestCase):
    def setUp(self):
        self.state = create_rotation_paper_state(
            symbols=["BTCUSDT", "ETHUSDT"],
            initial_capital=5000,
            interval="4h",
            config_hash="test"
        )
        self.prices = {
            "BTCUSDT": 100.0,
            "ETHUSDT": 50.0
        }
        self.config = candidate_config()

    def run_process(
        self,
        signal_time,
        target,
        acknowledge_gap=False
    ):
        frames = processed_frames(signal_time)
        targets = pd.Series(
            [target],
            index=pd.DatetimeIndex([signal_time]),
            dtype="object"
        )

        with patch(
            "rotation_paper_trader.generate_rotation_targets",
            return_value=(frames, targets)
        ):
            return process_rotation_state(
                original_state=self.state,
                original_frames=frames,
                current_prices=self.prices,
                strategy_config=self.config,
                interval="4h",
                fee_rate=0.001,
                slippage_rate=0.0005,
                event_time="2026-08-24T00:01:00+00:00",
                acknowledge_gap=acknowledge_gap
            )

    def test_first_run_only_initializes_state(self):
        state, events, status, _, _ = self.run_process(
            pd.Timestamp("2026-08-24 00:00", tz="UTC"),
            "BTCUSDT"
        )

        self.assertEqual(status, "initialized")
        self.assertEqual(events, [])
        self.assertIsNone(state["portfolio"]["held_symbol"])

    def test_next_bar_buys_once_and_duplicate_does_not_buy_again(self):
        initial_time = pd.Timestamp("2026-08-24 00:00", tz="UTC")
        self.state["last_signal_time"] = initial_time.isoformat()
        next_time = initial_time + pd.Timedelta(hours=4)
        state, events, status, _, _ = self.run_process(
            next_time,
            "BTCUSDT"
        )

        self.assertEqual(status, "processed")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "buy")
        self.assertAlmostEqual(events[0]["exposure"], 0.50)

        self.state = state
        duplicate_state, duplicate_events, duplicate_status, _, _ = (
            self.run_process(next_time, "BTCUSDT")
        )

        self.assertEqual(duplicate_status, "duplicate")
        self.assertEqual(duplicate_events, [])
        self.assertEqual(
            duplicate_state["portfolio"]["quantity"],
            state["portfolio"]["quantity"]
        )

    def test_missed_bar_is_rejected(self):
        initial_time = pd.Timestamp("2026-08-24 00:00", tz="UTC")
        self.state["last_signal_time"] = initial_time.isoformat()

        with self.assertRaises(PaperDataGapError):
            self.run_process(
                initial_time + pd.Timedelta(hours=8),
                "BTCUSDT"
            )

    def test_acknowledged_gap_skips_old_bars_without_trading(self):
        initial_time = pd.Timestamp("2026-08-24 00:00", tz="UTC")
        latest_time = initial_time + pd.Timedelta(hours=12)
        self.state["last_signal_time"] = initial_time.isoformat()
        original_portfolio = self.state["portfolio"].copy()

        state, events, status, _, _ = self.run_process(
            latest_time,
            "ETHUSDT",
            acknowledge_gap=True
        )

        self.assertEqual(status, "gap_acknowledged_no_trade")
        self.assertEqual(state["last_signal_time"], latest_time.isoformat())
        self.assertEqual(state["portfolio"], original_portfolio)
        self.assertEqual(state["gap_acknowledgement_count"], 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "data_gap_acknowledged")
        self.assertEqual(events[0]["missed_bars"], 2)


if __name__ == "__main__":
    unittest.main()
