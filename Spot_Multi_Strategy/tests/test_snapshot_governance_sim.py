"""Synthetic vote and price fixtures; no real outcome is read."""

from datetime import datetime, timezone
from unittest.mock import patch
import unittest

import pandas as pd

from research.snapshot_governance_source import END, SPACES, START, Proposal
from research.snapshot_governance_sim import run, signals


def seconds(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def proposal(ident: str, end_text: str, votes: int, *, updated_late: bool = False) -> Proposal:
    end = seconds(end_text)
    return Proposal(ident, "UNIUSDT", SPACES["UNIUSDT"], end - 3 * 86400,
                    end - 2 * 86400, end, votes,
                    end + (2 * 86400 if updated_late else 3600))


class GovernanceSimulationTests(unittest.TestCase):
    def test_six_prior_same_end_and_availability(self):
        warmup = [proposal(str(i), f"2020-{i+1:02d}-10", 10) for i in range(6)]
        event = proposal("signal", "2021-06-10", 25)
        simultaneous = proposal("same_end", "2021-06-10", 100)
        late = proposal("late", "2021-06-20", 100, updated_late=True)
        entries, audit = signals(warmup + [event, simultaneous, late])
        self.assertEqual(entries, {datetime(2021, 6, 12).date(): ("UNIUSDT",)})
        self.assertEqual(audit["source_ineligible"], 1)
        self.assertEqual(audit["high_turnout_events"], 2)

    def test_flat_prices_lose_after_fees_and_do_not_claim_gate(self):
        warmup = [proposal(str(i), f"2020-{i+1:02d}-10", 10) for i in range(6)]
        event = proposal("signal", "2022-06-10", 100)
        terminal_event = proposal("terminal", "2025-12-23", 100)
        prices = {}
        for symbol in (*SPACES, "BTCUSDT"):
            first = "2020-10-01" if symbol == "BTCUSDT" else str(START)
            index = pd.date_range(first, END - pd.Timedelta(days=1), tz="UTC", freq="D")
            prices[symbol] = pd.DataFrame(
                {name: [100.0] * len(index) for name in ("open", "high", "low", "close")},
                index=index)
        with patch("research.snapshot_governance_sim._bootstrap",
                   return_value={"joint_gate_fraction": 0.0, "paths": 5000}):
            result = run(warmup + [event, terminal_event], prices,
                         {symbol: 100.0 for symbol in prices})
        self.assertEqual(result["closed_trades"], 2)
        self.assertLess(result["terminal_equity"], 10_000)
        self.assertFalse(result["numerical_gate"])
        self.assertFalse(result["full_gate"])
        self.assertEqual(result["median_holding_days"], 7)


if __name__ == "__main__":
    unittest.main()
