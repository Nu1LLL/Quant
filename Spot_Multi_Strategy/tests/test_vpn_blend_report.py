import unittest

import numpy as np
import pandas as pd

import vpn_blend_report


class VpnBlendReportTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(
            vpn_blend_report.START, vpn_blend_report.END, freq="B"
        )
        self.existing = pd.DataFrame({
            "TQQQ_WTMF_SWITCH": 0.0005 + 0.001 * np.sin(np.arange(len(self.index))),
            "BTC_TSMOM30_DD": 0.0007 + 0.002 * np.cos(np.arange(len(self.index))),
        }, index=self.index)
        self.vpn = pd.Series(
            100 * np.exp(np.linspace(0, 1, len(self.index))), index=self.index
        )

    def test_requires_full_preregistered_window(self):
        with self.assertRaises(ValueError):
            vpn_blend_report.build_components(self.existing, self.vpn.iloc[1:])

    def test_vpn_return_uses_only_past_and_present_levels(self):
        before = vpn_blend_report.build_components(self.existing, self.vpn)
        changed = self.vpn.copy()
        changed.iloc[1500:] *= 2
        after = vpn_blend_report.build_components(self.existing, changed)
        pd.testing.assert_series_equal(before["VPN"].iloc[:1500], after["VPN"].iloc[:1500])

    def test_has_seven_fixed_scenarios(self):
        summary, artifacts, components = vpn_blend_report.run_experiment(
            self.existing, self.vpn, repetitions=39
        )
        expected = {
            "VPN_1x", "VPN_2x", "VPN_3x", "BLEND_1x", "BLEND_2x",
            "BLEND_3x", "BLEND_RISK_OVERLAY",
        }
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertEqual(
            list(components),
            ["TQQQ_WTMF_SWITCH", "BTC_TSMOM30_DD", "VPN"]
        )
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
