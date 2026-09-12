import unittest

import numpy as np
import pandas as pd

import bybit_funding_dispersion_oos as strategy
import bybit_funding_dispersion_oos_report as report


class FakeResponse:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        return None

    def json(self):
        return {"retCode": 0, "retMsg": "OK", "result": {"list": self.rows}}


class FakeSession:
    def __init__(self, rows):
        self.rows = rows

    def get(self, *_args, **_kwargs):
        return FakeResponse(self.rows)


class BybitFundingDispersionTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-09-01", "2026-09-11", freq="8h", tz="UTC")
        self.frames = {}
        for number, symbol in enumerate(report.SYMBOLS):
            n = len(self.index)
            self.frames[symbol] = pd.DataFrame({
                "funding_time": self.index,
                "funding_rate": (number - 1.5) * 0.0001,
                "mark_price": 100 * np.exp(np.linspace(0, 0.2 + number * 0.02, n)),
            })

    def test_signal_is_lagged_and_market_neutral(self):
        panel = strategy.build_common_panel(self.frames)
        weights, _ = strategy.lagged_rank_weights(panel)
        self.assertTrue(weights.iloc[0].isna().all())
        self.assertAlmostEqual(float(weights.iloc[1].sum()), 0.0)
        self.assertAlmostEqual(float(weights.iloc[1].abs().sum()), 1.0)

    def test_string_millisecond_funding_timestamp_is_parsed(self):
        timestamp = pd.Timestamp("2026-09-10 16:00:00", tz="UTC")
        rows = [{
            "fundingRateTimestamp": str(int(timestamp.timestamp() * 1000)),
            "fundingRate": "0.0001",
        }]
        result = strategy.download_funding(
            "BTCUSDT", "2026-09-10", "2026-09-11", session=FakeSession(rows)
        )
        self.assertEqual(result.iloc[0]["funding_time"], timestamp)

    def test_four_hour_mark_close_maps_to_funding_boundary(self):
        candle_start = pd.Timestamp("2026-09-10 12:00:00", tz="UTC")
        rows = [[str(int(candle_start.timestamp() * 1000)), "99", "101", "98", "100"]]
        result = strategy.download_mark_klines(
            "BTCUSDT", "2026-09-10", "2026-09-11", session=FakeSession(rows)
        )
        self.assertEqual(
            result.iloc[0]["funding_time"],
            pd.Timestamp("2026-09-10 16:00:00", tz="UTC"),
        )
        self.assertEqual(float(result.iloc[0]["mark_price"]), 100.0)

    def test_current_funding_change_does_not_change_current_position(self):
        panel = strategy.build_common_panel(self.frames)
        before, _ = strategy.lagged_rank_weights(panel)
        changed = panel.copy()
        changed.iloc[100, changed.columns.get_loc("BTCUSDT__funding")] = 1.0
        after, _ = strategy.lagged_rank_weights(changed)
        pd.testing.assert_series_equal(before.iloc[100], after.iloc[100])

    def test_costs_financing_and_fixed_scenarios(self):
        panel = strategy.build_common_panel(self.frames)
        _, detail, _ = strategy.run_notional(panel, 2.0)
        self.assertTrue((detail["trading_cost"] >= 0).all())
        self.assertAlmostEqual(float(detail["financing_cost"].iloc[0]), 0.04 / 1095)
        _, results, mc, _, _ = report.run_experiment(self.frames, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)

    def test_future_mutation_does_not_change_past_return(self):
        panel = strategy.build_common_panel(self.frames)
        before, _, _ = strategy.run_notional(panel, 1.0)
        changed = panel.copy()
        changed.loc[changed.index[1000]:, "XRPUSDT__funding"] = 0.1
        after, _, _ = strategy.run_notional(changed, 1.0)
        pd.testing.assert_series_equal(before.loc[:before.index[332]], after.loc[:after.index[332]])

    def test_incomplete_oos_grid_is_rejected(self):
        changed = {key: value.copy() for key, value in self.frames.items()}
        missing = pd.Timestamp("2022-01-01 08:00:00", tz="UTC")
        changed["EOSUSDT"] = changed["EOSUSDT"][
            changed["EOSUSDT"]["funding_time"] != missing
        ]
        with self.assertRaises(ValueError):
            report.run_experiment(changed, simulations=20)


if __name__ == "__main__":
    unittest.main()
