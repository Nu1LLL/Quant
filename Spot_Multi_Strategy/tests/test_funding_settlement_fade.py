"""Synthetic event-clock, timing, capacity and futures-account checks."""

from __future__ import annotations

from datetime import date
import math
import unittest
from unittest.mock import patch

import numpy as np

from research import funding_settlement_fade_sim as sim
from research.funding_settlement_fade_source import (
    DAY_BARS, WARMUP_BARS, SourceFailure, Tape,
)


def tape_one_day() -> Tape:
    n = WARMUP_BARS + DAY_BARS
    row = np.array([100., 100.1, 99.9, 100.01, 1_000_000.])
    prices = {"BTCUSDT": np.tile(row, (n, 1)),
              "ETHUSDT": np.tile(row, (n, 1))}
    events = frozenset(WARMUP_BARS + slot * 96 for slot in range(3))
    return Tape(prices, {symbol: events for symbol in prices}, {})


class SettlementFadeTest(unittest.TestCase):
    def test_causal_signal_and_exact_net_account(self) -> None:
        tape = tape_one_day()
        i = WARMUP_BARS
        tape.prices["BTCUSDT"][i, 2:4] = (98.9, 99.)
        tape.prices["BTCUSDT"][i + 14, :4] = (101., 101.1, 99.9, 100.01)
        with patch.object(sim, "EVAL_DAYS", 1):
            events = sim.plan_events(tape)
            self.assertEqual([x.side["BTCUSDT"] for x in events], [1, 0, 0])
            self.assertEqual([x.side["ETHUSDT"] for x in events], [0, 0, 0])
            # A future settlement's close cannot change the current plan.
            tape.prices["BTCUSDT"][i + 96, 2:4] = (90., 90.1)
            self.assertEqual(sim.plan_events(tape)[0].side, events[0].side)
            result = sim.simulate(tape, events)
        q = math.floor((.5 * sim.INITIAL / (100. * (1 + sim.SLIP))) / .001) * .001
        entry = 100. * (1 + sim.SLIP)
        exit_price = 101. * (1 - sim.SLIP)
        expected = sim.INITIAL + q * (exit_price - entry) - q * (entry + exit_price) * sim.FEE
        self.assertAlmostEqual(result.terminal_cash, expected)
        self.assertEqual(result.event_entries, 1)
        self.assertEqual([f["bar"] for f in result.fills], [i + 2, i + 14])
        self.assertLess(result.days[0].low, sim.INITIAL)

    def test_funding_while_open_fails(self) -> None:
        tape = tape_one_day()
        i = WARMUP_BARS
        event = sim.Event(date(2021, 1, 1), i, {"BTCUSDT": 1, "ETHUSDT": 0},
                          {"BTCUSDT": 1_000_000., "ETHUSDT": 1_000_000.})
        # This synthetic event violates the no-funding-while-held rule.
        tape.funding_index["BTCUSDT"] = frozenset((*tape.funding_index["BTCUSDT"], i + 5))
        with patch.object(sim, "EVAL_DAYS", 1):
            with self.assertRaisesRegex(SourceFailure, "funding while"):
                sim.simulate(tape, (event, event, event))

    def test_capacity_fails_closed(self) -> None:
        tape = tape_one_day()
        i = WARMUP_BARS
        events = tuple(sim.Event(date(2021, 1, 1), i + slot * 96,
                                 {"BTCUSDT": 1 if slot == 0 else 0,
                                  "ETHUSDT": 0},
                                 {"BTCUSDT": 4_990., "ETHUSDT": 1_000_000.})
                       for slot in range(3))
        with patch.object(sim, "EVAL_DAYS", 1):
            with self.assertRaisesRegex(SourceFailure, "capacity"):
                sim.simulate(tape, events)


if __name__ == "__main__":
    unittest.main()
