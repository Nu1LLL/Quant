"""Synthetic-only checks for the frozen spot-flow accounting engine."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.spot_flow_backtest import (  # noqa: E402
    SYMBOLS, BacktestIntegrityError, _run_period, _signals,
    block_monte_carlo, run_backtest,
)


START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2024-01-04", tz="UTC")


def fixture(*, flows=None, prices=None, funding_rates=None, start=START, end=END):
    grid = pd.date_range(start, end, freq="4h", inclusive="left")
    event_grid = pd.date_range(start, end, freq="8h", inclusive="left")
    flows = flows or {}
    prices = prices or {}
    funding_rates = funding_rates or {}
    spot = {}
    trade = {}
    mark = {}
    funding = {}
    for symbol in SYMBOLS:
        volume = np.full(len(grid), 1000.0)
        buy = np.array([500.0 + 400.0 * flows.get((symbol, time.floor("D")), 0.0)
                        for time in grid])
        spot[symbol] = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                                     "close": 100.0, "quote_volume": volume,
                                     "taker_buy_quote_volume": buy}, index=grid)
        px = np.array([prices.get((symbol, time), 100.0) for time in grid], dtype=float)
        trade[symbol] = pd.DataFrame({"open": px, "high": px, "low": px,
                                      "close": px, "quote_volume": volume}, index=grid)
        mark[symbol] = pd.DataFrame({"open": px, "high": px, "low": px,
                                     "close": px}, index=grid)
        funding[symbol] = pd.DataFrame({"funding_time": event_grid,
                                        "rate": [funding_rates.get((symbol, time), 0.0)
                                                 for time in event_grid],
                                        "funding_interval_hours": 8})
    return SimpleNamespace(spot_bars=spot, trade_bars=trade,
                           mark_bars=mark, funding_events=funding, manifest={})


class SpotFlowBacktestTests(unittest.TestCase):
    def test_prior_day_ranking_and_alphabetical_tie(self):
        bundle = fixture()
        decisions = _signals(bundle.spot_bars, START, END)
        self.assertEqual(decisions[START + pd.Timedelta(days=1, hours=4)],
                         ("BTCUSDT", "XRPUSDT"))

    def test_flow_day_does_not_change_d_plus_one_first_order(self):
        day1 = START
        day2 = START + pd.Timedelta(days=1)
        bundle = fixture(flows={("BTCUSDT", day1): .9, ("LTCUSDT", day1): -.9,
                                ("BTCUSDT", day2): -.9, ("LTCUSDT", day2): .9})
        decisions = _signals(bundle.spot_bars, START, END)
        self.assertEqual(decisions[day2 + pd.Timedelta(hours=4)],
                         ("BTCUSDT", "LTCUSDT"))
        # Information from day 2 is allowed only in day 3's order.
        self.assertEqual(decisions[day2 + pd.Timedelta(days=1, hours=4)],
                         ("LTCUSDT", "BTCUSDT"))

    def test_actual_entry_and_terminal_exit_turnover_and_costs(self):
        day1 = START
        bundle = fixture(flows={("BTCUSDT", day1): .9, ("LTCUSDT", day1): -.9})
        result = _run_period(bundle, start=START, end=END)
        costs = result["diagnostics"]["accounting_usd"]
        self.assertAlmostEqual(costs["traded_notional_usd"], 20_000.0)
        self.assertAlmostEqual(costs["taker_fees_usd"], 10.0)
        self.assertAlmostEqual(costs["slippage_usd"], 20.0)
        self.assertAlmostEqual(result["equity"][-1]["equity_usd"], 9_970.0)
        self.assertEqual(len(result["orders"]), 4)
        self.assertEqual(len(result["basket_cycles"]), 1)
        self.assertAlmostEqual(result["basket_cycles"][0]["net_pnl_usd"], -30.0)

    def test_funding_uses_actual_event_and_signed_holding(self):
        day1 = START
        events = pd.date_range(START, END, freq="8h", inclusive="left")
        rates = {( "BTCUSDT", event): .01 for event in events}
        bundle = fixture(flows={("BTCUSDT", day1): .9, ("LTCUSDT", day1): -.9},
                         funding_rates=rates)
        result = _run_period(bundle, start=START, end=END)
        self.assertAlmostEqual(result["diagnostics"]["accounting_usd"]["funding_usd"], -150.0)
        self.assertAlmostEqual(result["equity"][-1]["equity_usd"], 9_820.0)

    def test_funding_exactly_at_04_precedes_terminal_liquidation(self):
        bundle = fixture(flows={("BTCUSDT", START): .9,
                                ("LTCUSDT", START): -.9})
        terminal = END - pd.Timedelta(hours=20)
        frame = bundle.funding_events["BTCUSDT"]
        boundary_event = pd.DataFrame({"funding_time": [terminal],
                                       "rate": [.01],
                                       "funding_interval_hours": [4]})
        bundle.funding_events["BTCUSDT"] = (
            pd.concat((frame, boundary_event), ignore_index=True)
            .sort_values("funding_time").reset_index(drop=True)
        )
        result = _run_period(bundle, start=START, end=END)
        self.assertAlmostEqual(result["diagnostics"]["accounting_usd"]["funding_usd"], -50.0)
        self.assertAlmostEqual(result["equity"][-1]["equity_usd"], 9_920.0)

    def test_price_pnl_and_two_sided_turnover_costs_reconcile(self):
        terminal = END - pd.Timedelta(hours=20)
        bundle = fixture(flows={("BTCUSDT", START): .9,
                                ("LTCUSDT", START): -.9},
                         prices={("BTCUSDT", terminal): 110.0,
                                 ("LTCUSDT", terminal): 90.0})
        result = _run_period(bundle, start=START, end=END)
        self.assertAlmostEqual(result["equity"][-1]["equity_usd"], 10_970.0)
        self.assertAlmostEqual(sum(lot["net_pnl_usd"] for lot in result["closed_lots"]), 970.0)

    def test_intrabar_adverse_long_low_and_short_high(self):
        day1 = START
        bundle = fixture(flows={("BTCUSDT", day1): .9, ("LTCUSDT", day1): -.9})
        t = START + pd.Timedelta(days=1, hours=8)
        bundle.mark_bars["BTCUSDT"].at[t, "low"] = 80.0
        bundle.mark_bars["LTCUSDT"].at[t, "high"] = 120.0
        result = _run_period(bundle, start=START, end=END)
        self.assertLess(result["diagnostics"]["full"]["max_drawdown_intrabar"], -.20)

    def test_rejects_missing_bar_and_invalid_taker_volume(self):
        bundle = fixture()
        bundle.spot_bars["BTCUSDT"] = bundle.spot_bars["BTCUSDT"].iloc[1:]
        with self.assertRaisesRegex(BacktestIntegrityError, "incomplete 4h grid"):
            _run_period(bundle, start=START, end=END)
        bundle = fixture()
        bundle.spot_bars["BTCUSDT"].iloc[0,
            bundle.spot_bars["BTCUSDT"].columns.get_loc("taker_buy_quote_volume")] = 1001.0
        with self.assertRaisesRegex(BacktestIntegrityError, "invalid taker"):
            _run_period(bundle, start=START, end=END)

    def test_block_monte_carlo_is_deterministic_and_keeps_zero_days(self):
        returns = pd.Series([0.0] * 25 + [.01] * 15 + [-.02] * 10)
        first = block_monte_carlo(returns, block_days=21, paths=64, seed=20261102)
        second = block_monte_carlo(returns, block_days=21, paths=64, seed=20261102)
        self.assertEqual(first, second)
        self.assertEqual(first["paths"], 64)
        self.assertEqual(first["block_days"], 21)

    def test_public_formal_entry_rejects_unfrozen_cost(self):
        with self.assertRaisesRegex(ValueError, "formal rule fixes"):
            run_backtest(None, fee_bps=4)


if __name__ == "__main__":
    unittest.main()
