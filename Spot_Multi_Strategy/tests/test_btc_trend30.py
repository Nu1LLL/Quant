import unittest

import numpy as np
import pandas as pd

import btc_trend30


class BtcTrend30Tests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(
            "2016-08-01", periods=1500, freq="D", tz="UTC"
        )
        self.price = pd.Series(
            100 * np.exp(np.linspace(0, 1.0, len(self.index))),
            index=self.index,
        )

    def test_future_mutation_does_not_change_past_position(self):
        before = btc_trend30.build_positions(self.price, "TSMOM30_DD")
        changed = self.price.copy()
        changed.iloc[1000:] *= 0.2
        after = btc_trend30.build_positions(changed, "TSMOM30_DD")
        pd.testing.assert_series_equal(before.iloc[:1000], after.iloc[:1000])

    def test_drawdown_overlay_halves_active_position_next_day(self):
        price = self.price.copy()
        price.iloc[100:] = price.iloc[100:] * 0.5
        plain = btc_trend30.build_positions(price, "TSMOM30")
        protected = btc_trend30.build_positions(price, "TSMOM30_DD")
        active = (plain > 0) & (price / price.cummax() - 1 <= -0.15).shift(1).fillna(False)
        if active.any():
            np.testing.assert_allclose(
                protected[active].to_numpy(), plain[active].to_numpy() * 0.5
            )

    def test_buy_hold_position_is_shifted_once(self):
        position = btc_trend30.build_positions(self.price, "BUY_HOLD")
        self.assertEqual(position.iloc[0], 0.0)
        self.assertTrue((position.iloc[1:] == 1.0).all())

    def test_backtest_charges_initial_entry(self):
        simulation, yearly, metrics = btc_trend30.run_backtest(
            self.price, "BUY_HOLD", start="2016-09-12", end="2019-09-10"
        )
        self.assertAlmostEqual(simulation["turnover"].iloc[0], 1.0)
        self.assertGreater(simulation["trading_cost"].iloc[0], 0.0)
        self.assertIn("worst_rolling_3y_sharpe", metrics)


if __name__ == "__main__":
    unittest.main()
