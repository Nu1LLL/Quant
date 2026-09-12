import unittest

import numpy as np
import pandas as pd

import smallcap_macro_oos
import smallcap_macro_oos_report


class SmallcapMacroOosTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2009-01-01", "2026-09-10", freq="B", tz="UTC")
        n = len(self.index)
        self.prices = {
            "IWM": pd.Series(100*np.exp(np.linspace(0, 1, n)), index=self.index),
            "TNA": pd.Series(100*np.exp(np.linspace(0, 2, n)), index=self.index),
            "TBT": pd.Series(100*np.exp(np.linspace(0, .2, n)), index=self.index),
            "UGL": pd.Series(100*np.exp(np.linspace(0, .5, n)), index=self.index),
        }
        self.vix = pd.Series(15.0, index=self.index)
        self.vix3m = pd.Series(18.0, index=self.index)

    def test_positions_are_exclusive_and_defensive_is_equal(self):
        p = smallcap_macro_oos.build_positions(**{k.lower(): v for k, v in self.prices.items()})
        self.assertTrue(p.sum(axis=1).isin([0.0, 1.0]).all())
        self.assertTrue((p["TBT"] == p["UGL"]).all())

    def test_future_mutation_does_not_change_past(self):
        args = {k.lower(): v for k, v in self.prices.items()}
        before = smallcap_macro_oos.build_positions(**args)
        changed = args.copy(); changed["iwm"] = args["iwm"].copy(); changed["iwm"].iloc[3000:] *= 2
        after = smallcap_macro_oos.build_positions(**changed)
        pd.testing.assert_frame_equal(before.iloc[:3000], after.iloc[:3000])

    def test_report_keeps_preregistered_oos_and_validation(self):
        dev, oos, validation, mc, regimes = smallcap_macro_oos_report.run_experiment(
            self.prices, self.vix, self.vix3m, monte_carlo_simulations=50
        )
        self.assertEqual(oos[0]["open_time"].iloc[0], smallcap_macro_oos_report.OOS_START)
        self.assertTrue(validation["checks"]["independent_oos"])
        self.assertTrue(validation["checks"]["costs_included"])
        self.assertEqual(mc["simulations"], 50)
        self.assertEqual(set(regimes["family"]), {"trade_state", "vix_curve"})


if __name__ == "__main__":
    unittest.main()
