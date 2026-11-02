import unittest

import pandas as pd

import market_execution_costs as costs


class MarketExecutionCostTests(unittest.TestCase):
    @staticmethod
    def base():
        return pd.DataFrame({"best_bid": [99.95], "best_ask": [100.05],
                             "financing_bps_horizon": [0.2]})

    def test_equity_per_share_commission(self):
        frame = self.base().assign(commission_per_unit_each_side=0.005,
                                   impact_bps_each_side=0.25)
        self.assertAlmostEqual(costs.round_trip_cost_bps(frame, "stock").iloc[0], 11.7)

    def test_option_contract_cost(self):
        frame = self.base().assign(commission_per_contract_each_side=0.65,
                                   contract_multiplier=100, tick_size=0.01,
                                   slippage_ticks_each_side=1)
        expected = 10 + 2 * 0.65 / (100 * 100) * 1e4 + 2 * 0.01 / 100 * 1e4 + 0.2
        self.assertAlmostEqual(costs.round_trip_cost_bps(frame, "option").iloc[0], expected)

    def test_futures_contract_cost(self):
        frame = self.base().assign(commission_per_contract_each_side=2.5,
                                   contract_multiplier=50, tick_size=0.25,
                                   slippage_ticks_each_side=1)
        expected = 10 + 2 * 2.5 / (100 * 50) * 1e4 + 2 * 0.25 / 100 * 1e4 + 0.2
        self.assertAlmostEqual(costs.round_trip_cost_bps(frame, "futures").iloc[0], expected)

    def test_fx_per_million_commission(self):
        frame = self.base().assign(commission_per_million_each_side=35,
                                   impact_bps_each_side=0.1)
        self.assertAlmostEqual(costs.round_trip_cost_bps(frame, "fx").iloc[0], 11.1)

    def test_crypto_negative_funding_cannot_erase_execution(self):
        frame = self.base().assign(fee_bps_each_side=2,
                                   impact_bps_each_side=0.5,
                                   financing_bps_horizon=-100)
        self.assertAlmostEqual(costs.round_trip_cost_bps(frame, "crypto").iloc[0], 15.0)

    def test_missing_or_invalid_costs_are_rejected(self):
        with self.assertRaises(ValueError):
            costs.round_trip_cost_bps(self.base(), "equity")
        frame = self.base().assign(commission_per_unit_each_side=-0.01,
                                   impact_bps_each_side=0.1)
        with self.assertRaises(ValueError):
            costs.round_trip_cost_bps(frame, "equity")


if __name__ == "__main__":
    unittest.main()
