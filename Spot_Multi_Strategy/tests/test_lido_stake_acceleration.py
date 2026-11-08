"""Synthetic causality and spot cash-account checks only."""

from datetime import date, timedelta
import math
import unittest

import numpy as np
import pandas as pd

from research.lido_stake_sim import simulate, weekly_targets


def fixture():
    first = date(2020, 12, 20)
    stake = {first + timedelta(days=7 * i):
             1000 * math.exp(.005 * i + .01 * math.sin(i * .17))
             for i in range(263)}
    index = pd.date_range("2020-11-01", "2025-12-31", tz="UTC", freq="D")
    prices = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                           "close": 100.0, "volume": 1_000_000.0}, index=index)
    return stake, prices, 100.0


class LidoStakeAccelerationTest(unittest.TestCase):
    def test_no_future_stake_can_change_first_signal(self):
        stake, _, _ = fixture()
        targets = weekly_targets(stake)
        self.assertEqual(min(targets), date(2021, 2, 18))
        first = min(targets)
        stake[date(2021, 2, 21)] *= 100
        self.assertEqual(weekly_targets(stake)[first], targets[first])
        self.assertIn(True, targets.values())
        self.assertIn(False, targets.values())

    def test_costs_and_terminal_exit_reconcile_without_borrowing(self):
        stake, prices, terminal = fixture()
        net = simulate(stake, prices, terminal)
        gross = simulate(stake, prices, terminal, slip=0.0, fee_rate=0.0)
        self.assertEqual(len(net.days), 1826)
        self.assertGreater(net.entries, 0)
        self.assertGreater(net.fees, 0)
        self.assertLess(net.final, gross.final)
        self.assertAlmostEqual(net.days[-1].end, net.final)
        self.assertTrue(all(t["day"] >= "2021-02-18" for t in net.trades))
        self.assertTrue(all(t["day"] <= "2026-01-01" for t in net.trades))
        self.assertAlmostEqual(net.trades[-1]["notional"] * .001,
                               net.trades[-1]["fee"])


if __name__ == "__main__":
    unittest.main()
