import unittest

import numpy as np
import pandas as pd

import microstructure_blackbox as blackbox


class MicrostructureBlackBoxTests(unittest.TestCase):
    @staticmethod
    def frame(rows=1800):
        rng = np.random.default_rng(33)
        index = pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC")
        imbalance = rng.uniform(-1, 1, rows)
        returns = 0.00008 * np.sign(imbalance) * imbalance ** 2 + rng.normal(0, 0.0002, rows)
        mid = 100 * np.exp(np.cumsum(returns))
        spread = 0.01
        bid_size = 100 * (1 + imbalance)
        ask_size = 100 * (1 - imbalance)
        intensity = rng.lognormal(0.0, 0.4, rows)
        return pd.DataFrame({
            "best_bid": mid - spread / 2, "best_ask": mid + spread / 2,
            "bid_size_1": bid_size, "ask_size_1": ask_size,
            "bid_depth_5": bid_size * 4, "ask_depth_5": ask_size * 4,
            "bid_depth_20": bid_size * 12, "ask_depth_20": ask_size * 12,
            "buy_notional": 1000 * intensity * (1 + imbalance),
            "sell_notional": 1000 * intensity * (1 - imbalance),
            "bid_cancel_notional": 100 * (1 - imbalance),
            "ask_cancel_notional": 100 * (1 + imbalance),
        }, index=index)

    def test_features_are_prefix_invariant(self):
        frame = self.frame()
        prefix = blackbox.build_features(frame.iloc[:1200])
        extended = blackbox.build_features(frame)
        pd.testing.assert_frame_equal(prefix, extended.loc[prefix.index])

    def test_order_flow_imbalance_matches_price_queue_rules(self):
        frame = self.frame(4)
        frame["best_bid"] = [100.00, 100.00, 100.01, 100.00]
        frame["best_ask"] = [100.02, 100.02, 100.03, 100.02]
        frame["bid_size_1"] = [10.0, 13.0, 7.0, 11.0]
        frame["ask_size_1"] = [12.0, 8.0, 9.0, 10.0]
        result = blackbox.order_flow_imbalance(frame)
        self.assertAlmostEqual(result["ofi_l1"].iloc[1], 7.0)
        self.assertAlmostEqual(result["ofi_l1"].iloc[2], 15.0)
        self.assertAlmostEqual(result["ofi_l1"].iloc[3], -17.0)

    def test_v1_and_v2_feature_sets_remain_reproducible(self):
        frame = self.frame()
        v1, _ = blackbox.train_and_predict(
            frame.iloc[:1200], frame.iloc[1200:], feature_version="v1"
        )
        v2, _ = blackbox.train_and_predict(
            frame.iloc[:1200], frame.iloc[1200:], feature_version="v2"
        )
        self.assertEqual(len(v2.kernel_feature_columns), len(v1.kernel_feature_columns) + 3)

    def test_blackbox_outputs_auditable_capped_orders(self):
        frame = self.frame()
        model, prediction = blackbox.train_and_predict(frame.iloc[:1200], frame.iloc[1200:])
        orders = blackbox.generate_orders(
            prediction, frame, model, venue="synthetic", symbol="TEST"
        )
        expected = {"action", "probability_up", "expected_return_bps",
                    "cost_gate_bps", "hmm_stress_probability", "kelly_position"}
        self.assertTrue(expected.issubset(orders.columns))
        self.assertLessEqual(orders["kelly_position"].abs().max(), 0.25 + 1e-12)
        self.assertTrue(set(orders["action"]).issubset({"BUY", "SELL", "FLAT"}))

    def test_first_test_quote_cannot_change_fitted_model(self):
        frame = self.frame()
        train, test = frame.iloc[:1200], frame.iloc[1200:].copy()
        original, _ = blackbox.train_and_predict(train, test)
        test.iloc[0, test.columns.get_loc("best_bid")] *= 1.01
        test.iloc[0, test.columns.get_loc("best_ask")] *= 1.01
        changed, _ = blackbox.train_and_predict(train, test)
        np.testing.assert_allclose(original.kernel.return_beta, changed.kernel.return_beta)
        np.testing.assert_allclose(original.kernel.class_beta, changed.kernel.class_beta)
        np.testing.assert_allclose(original.hmm.means, changed.hmm.means)

    def test_higher_explicit_cost_cannot_increase_trades(self):
        frame = self.frame()
        model, prediction = blackbox.train_and_predict(frame.iloc[:1200], frame.iloc[1200:])
        test_frame = frame.reindex(prediction.index)
        spread = (test_frame["best_ask"] - test_frame["best_bid"]) / (
            (test_frame["best_ask"] + test_frame["best_bid"]) / 2
        ) * 1e4
        low = blackbox.generate_orders(
            prediction, frame, model, "synthetic", "TEST", explicit_cost_bps=spread
        )
        high = blackbox.generate_orders(
            prediction, frame, model, "synthetic", "TEST",
            explicit_cost_bps=spread + 1000,
        )
        self.assertLessEqual(high["action"].ne("FLAT").sum(), low["action"].ne("FLAT").sum())
        with self.assertRaises(ValueError):
            blackbox.generate_orders(
                prediction, frame, model, "synthetic", "TEST",
                explicit_cost_bps=spread - 0.01,
            )

    def test_alpha_health_disables_negative_edge(self):
        rng = np.random.default_rng(4)
        negative = pd.Series(rng.normal(-0.001, 0.001, 1000))
        positive = pd.Series(rng.normal(0.001, 0.001, 1000))
        self.assertEqual(blackbox.alpha_health(negative)["status"], "DISABLED")
        self.assertEqual(blackbox.alpha_health(positive)["status"], "ACTIVE")

    def test_point_in_time_universe_uses_market_specific_thresholds(self):
        snapshot = pd.DataFrame([
            {"market": "crypto", "symbol": "BTC", "notional_30d": 1e9,
             "spread_bps_30d": 1.0, "depth_usd_30d": 1e6, "history_days": 1000,
             "data_complete": True, "shortable": True},
            {"market": "crypto", "symbol": "ILLQ", "notional_30d": 1e5,
             "spread_bps_30d": 100.0, "depth_usd_30d": 10, "history_days": 1000,
             "data_complete": True, "shortable": True},
        ])
        selected = blackbox.point_in_time_universe(snapshot, {
            "crypto": {"min_notional": 1e7, "max_spread_bps": 5,
                       "min_depth": 1e5, "min_history_days": 90}
        })
        self.assertEqual(selected["symbol"].tolist(), ["BTC"])
