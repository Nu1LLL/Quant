import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

import dia_overnight_intraday_oos as strategy
import dia_overnight_intraday_oos_report as report


class DiaOvernightIntradayTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("1998-01-20", "2026-06-30", freq="B", tz="UTC")
        rng = np.random.default_rng(19980120)
        raw_close = 100 * np.cumprod(1 + rng.normal(0, 0.008, len(self.index)))
        raw_open = raw_close / (1 + rng.normal(0, 0.003, len(self.index)))
        factor = np.where(np.arange(len(self.index)) < 1000, 0.5, 1.0)
        self.frame = pd.DataFrame({
            "raw_open": raw_open,
            "raw_close": raw_close,
            "adjusted_close": raw_close * factor,
        }, index=self.index)

    def test_adjusted_open_uses_same_day_factor(self):
        components = strategy.return_components(self.frame)
        expected = self.frame.loc[components.index, "raw_open"] * (
            self.frame.loc[components.index, "adjusted_close"]
            / self.frame.loc[components.index, "raw_close"]
        )
        pd.testing.assert_series_equal(
            components["adjusted_open"], expected, check_names=False
        )

    def test_sequential_compounding_and_costs(self):
        returns, detail = strategy.run_scenario(self.frame, 1.0)
        first = detail.iloc[0]
        overnight_net = first["overnight_return"] - 0.0002
        intraday_net = -first["intraday_return"] - 0.0002 - 0.005 / 504
        self.assertAlmostEqual(float(returns.iloc[0]), (1 + overnight_net) * (1 + intraday_net) - 1)
        self.assertAlmostEqual(float(first["trading_cost"]), 0.0004)

    def test_future_mutation_does_not_change_past(self):
        baseline, _ = strategy.run_scenario(self.frame, 1.0)
        changed = self.frame.copy()
        changed.iloc[-1, changed.columns.get_loc("raw_close")] *= 0.8
        changed.iloc[-1, changed.columns.get_loc("adjusted_close")] *= 0.8
        rerun, _ = strategy.run_scenario(changed, 1.0)
        pd.testing.assert_series_equal(baseline.iloc[:-1], rerun.iloc[:-1])

    def test_report_has_fixed_scenarios_and_regimes(self):
        results, mc, regimes = report.run_experiment(self.frame, simulations=20)
        self.assertEqual(set(results), {1.0, 2.0, 3.0})
        self.assertEqual(mc["simulations"], 20)
        self.assertEqual(set(regimes["family"]), {"period", "event"})

    def test_short_history_and_large_gap_fail(self):
        with self.assertRaises(ValueError):
            report.run_experiment(self.frame.loc["2015":], simulations=20)
        gapped = self.frame.drop(self.frame.loc["2010-01-01":"2010-02-01"].index)
        with self.assertRaises(ValueError):
            strategy.validate_ohlc(gapped)

    def test_download_uses_fixed_dates_not_max_range(self):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"chart": {"result": [{
            "timestamp": [946684800, 946771200],
            "indicators": {
                "quote": [{"open": [10.0, 10.1], "close": [10.1, 10.2]}],
                "adjclose": [{"adjclose": [9.9, 10.0]}],
            },
        }]}}
        session = MagicMock()
        session.get.return_value = response
        strategy.download_ohlc(session=session)
        params = session.get.call_args.kwargs["params"]
        self.assertNotIn("range", params)
        self.assertEqual(params["interval"], "1d")
        self.assertEqual(params["period1"], int(strategy.REQUEST_START.timestamp()))


if __name__ == "__main__":
    unittest.main()
