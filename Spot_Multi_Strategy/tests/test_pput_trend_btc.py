import unittest

import numpy as np
import pandas as pd

import pput_trend_btc


class PputTrendBtcTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(
            "2016-09-12", periods=1200, freq="B", tz="UTC"
        )
        rng = np.random.default_rng(731)
        self.components = pd.DataFrame({
            "PPUT": rng.normal(0.0003, 0.008, len(self.index)),
            "BTC_TSMOM30_DD": rng.normal(0.0008, 0.025, len(self.index)),
        }, index=self.index)

    def test_pput_signal_only_updates_after_month_end(self):
        rising = np.linspace(100.0, 200.0, 260)
        falling = np.linspace(200.0, 80.0, 260)
        level = pd.Series(
            np.concatenate([rising, falling]), index=self.index[:520]
        )
        _, position, _ = pput_trend_btc.build_pput_trend_returns(level)
        changes = position.diff().fillna(position).ne(0.0)
        change_dates = position.index[changes]
        previous_dates = position.index[position.index.get_indexer(change_dates) - 1]
        self.assertGreater(len(change_dates), 0)
        for previous_date, change_date in zip(previous_dates, change_dates):
            self.assertNotEqual(previous_date.month, change_date.month)

    def test_pput_future_mutation_does_not_change_past(self):
        level = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 0.6, len(self.index))),
            index=self.index,
        )
        before = pput_trend_btc.build_pput_trend_returns(level)
        changed = level.copy()
        changed.iloc[900:] *= np.linspace(1.0, 3.0, len(changed) - 900)
        after = pput_trend_btc.build_pput_trend_returns(changed)
        for before_item, after_item in zip(before, after):
            pd.testing.assert_series_equal(
                before_item.iloc[:900], after_item.iloc[:900]
            )

    def test_active_blend_weights_are_nonnegative_and_sum_to_one(self):
        positions = pput_trend_btc.build_blend_positions(self.components)
        active = positions.sum(axis=1) > 0.0
        self.assertTrue((positions >= 0.0).all().all())
        np.testing.assert_allclose(
            positions.loc[active].sum(axis=1).to_numpy(), 1.0
        )

    def test_scenarios_report_costs_and_metrics(self):
        simulation, _, _, metrics = pput_trend_btc.run_static_scenario(
            self.components, leverage=2.0
        )
        self.assertTrue((simulation["financing_cost"] >= 0.0).all())
        self.assertIn("worst_rolling_3y_sharpe", metrics)
        overlay, _, _, overlay_metrics = (
            pput_trend_btc.run_overlay_scenario(self.components)
        )
        self.assertTrue((overlay["cost"] >= 0.0).all())
        self.assertIn("worst_rolling_3y_sharpe", overlay_metrics)


if __name__ == "__main__":
    unittest.main()
