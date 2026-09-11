import unittest

import numpy as np
import pandas as pd

import tqqq_trend_btc
import tqqq_trend_btc_report


class TqqqTrendBtcTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(
            "2015-09-01", periods=2900, freq="B", tz="UTC"
        )
        self.qqq = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 0.8, len(self.index))),
            index=self.index,
        )
        self.tqqq = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 2.0, len(self.index))),
            index=self.index,
        )
        btc_dates = pd.date_range(
            "2016-09-12", "2026-09-10", freq="D", tz="UTC"
        )
        self.btc = pd.DataFrame({
            "open_time": btc_dates,
            "equity": np.exp(np.linspace(0.0, 1.0, len(btc_dates))),
        })

    def test_position_updates_after_month_end_only(self):
        position = tqqq_trend_btc.build_tactical_position(self.qqq)
        changes = position.diff().fillna(position).ne(0.0)
        for date in position.index[changes]:
            previous = position.index[position.index.get_loc(date) - 1]
            self.assertNotEqual(previous.month, date.month)

    def test_future_mutation_does_not_change_past(self):
        before = tqqq_trend_btc.build_tactical_position(self.qqq)
        changed = self.qqq.copy()
        changed.iloc[1800:] *= np.linspace(1.0, 3.0, len(changed) - 1800)
        after = tqqq_trend_btc.build_tactical_position(changed)
        pd.testing.assert_series_equal(before.iloc[:1800], after.iloc[:1800])

    def test_sleeve_charges_initial_position(self):
        position = pd.Series(1.0, index=self.tqqq.index)
        simulation, _, metrics = tqqq_trend_btc.run_etf_sleeve(
            self.tqqq, position
        )
        self.assertAlmostEqual(simulation["turnover"].iloc[0], 1.0)
        self.assertGreater(metrics["total_trading_cost"], 0.0)

    def test_experiment_has_seven_fixed_scenarios(self):
        summary, artifacts, components, _ = tqqq_trend_btc_report.run_experiment(
            self.qqq, self.tqqq, self.btc, repetitions=99
        )
        expected = {
            "TQQQ_BUY_HOLD", "TACTICAL_TQQQ", "TACTICAL_RISK_OVERLAY",
            "BLEND_1x", "BLEND_2x", "BLEND_3x", "BLEND_RISK_OVERLAY",
        }
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertEqual(
            list(components), ["TACTICAL_TQQQ", "BTC_TSMOM30_DD"]
        )
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
