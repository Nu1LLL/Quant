import unittest

import numpy as np
import pandas as pd

import alpha_lifecycle as lifecycle


def make_returns(specifications, observations=500, seed=20260916):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=observations, freq="min", tz="UTC")
    rows = []
    shared = rng.normal(0, 0.001, observations)
    for market, venue, symbol, alpha_id, mean, correlated in specifications:
        noise = shared if correlated else rng.normal(0, 0.001, observations)
        for timestamp, value in zip(timestamps, noise + mean):
            rows.append({"timestamp": timestamp, "market": market, "venue": venue,
                         "symbol": symbol, "alpha_id": alpha_id,
                         "net_return": value})
    return pd.DataFrame(rows)


class AlphaLifecycleTests(unittest.TestCase):
    def test_positive_edge_activates_and_zero_edges_do_not_flood(self):
        specs = [("fx", "D", "EURUSD", "real", 0.00035, False)]
        specs += [("equity", "N", f"S{i}", f"null{i}", 0.0, False) for i in range(40)]
        result = lifecycle.evaluate_lifecycle(make_returns(specs, observations=800))
        real = result[result["alpha_id"].eq("real")].iloc[0]
        self.assertEqual(real["status"], "ACTIVE")
        null_active = result[result["alpha_id"].str.startswith("null")]["status"].eq("ACTIVE").sum()
        self.assertLessEqual(null_active, 2)

    def test_short_history_stays_shadow(self):
        data = make_returns([("future", "CME", "ES", "a", 0.001, False)], observations=199)
        result = lifecycle.evaluate_lifecycle(data)
        self.assertEqual(result.iloc[0]["status"], "SHADOW")
        self.assertEqual(result.iloc[0]["raw_weight"], 0.0)

    def test_recent_decay_disables_previous_edge(self):
        good = make_returns([("crypto", "X", "BTC", "a", 0.0004, False)], observations=500)
        bad = make_returns([("crypto", "X", "BTC", "a", -0.0007, False)], observations=500, seed=9)
        bad["timestamp"] += pd.Timedelta(days=2)
        config = lifecycle.LifecycleConfig(rolling_window=500)
        result = lifecycle.evaluate_lifecycle(pd.concat([good, bad]), config)
        self.assertEqual(result.iloc[0]["status"], "DISABLED")

    def test_allocation_respects_all_caps_and_clusters(self):
        specs = [
            ("option", "OPRA", "SPY-C1", "a", 0.0005, True),
            ("option", "OPRA", "SPY-C1", "b", 0.0005, True),
            ("option", "OPRA", "QQQ-C1", "c", 0.0005, False),
            ("fx", "D", "EURUSD", "d", 0.0005, False),
        ]
        data = make_returns(specs, observations=800)
        evaluated = lifecycle.evaluate_lifecycle(data)
        allocated = lifecycle.allocate_active_weights(evaluated, data)
        self.assertLessEqual(allocated["weight"].sum(), 1.0 + 1e-12)
        self.assertLessEqual(allocated.groupby("market")["weight"].sum().max(), 0.35 + 1e-12)
        self.assertLessEqual(
            allocated.groupby(["market", "venue", "symbol"])["weight"].sum().max(),
            0.20 + 1e-12,
        )
        self.assertLessEqual(allocated.groupby("cluster_id")["weight"].sum().max(), 0.20 + 1e-12)

    def test_input_order_does_not_change_results(self):
        data = make_returns([
            ("stock", "N", "A", "a", 0.0003, False),
            ("fx", "D", "EURUSD", "b", 0.0003, False),
        ])
        left = lifecycle.evaluate_lifecycle(data).sort_values(list(lifecycle.KEY_COLUMNS)).reset_index(drop=True)
        right = lifecycle.evaluate_lifecycle(data.sample(frac=1, random_state=8)).sort_values(list(lifecycle.KEY_COLUMNS)).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)


if __name__ == "__main__":
    unittest.main()
