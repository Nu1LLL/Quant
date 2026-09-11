import unittest

import numpy as np
import pandas as pd

import sgti_vxth_btc_report as report


class SgtiVxthBtcReportTests(unittest.TestCase):
    def test_parse_sgti_csv_requires_and_parses_columns(self):
        text = (
            "trade_date,SG Trend Indicator,SG CTA Index\n"
            "2024-01-02,100.0,100.0\n2024-01-03,101.0,100.5\n"
        )
        frame = report.parse_sgti_csv(text)
        self.assertEqual(len(frame), 2)
        self.assertEqual(str(frame["trade_date"].dt.tz), "UTC")
        with self.assertRaises(ValueError):
            report.parse_sgti_csv("date,value\n2024-01-02,100\n")

    def test_alignment_compounds_btc_between_common_dates(self):
        common = pd.to_datetime(["2024-01-05", "2024-01-08"], utc=True)
        sgti = pd.Series([100.0, 101.0], index=common)
        vxth = pd.Series([100.0, 102.0], index=common)
        btc_dates = pd.date_range("2024-01-05", "2024-01-08", freq="D", tz="UTC")
        btc = pd.DataFrame({
            "open_time": btc_dates, "equity": [1.0, 1.1, 1.21, 1.331]
        })
        returns = report.align_components(sgti, vxth, btc)
        self.assertAlmostEqual(returns["BTC_TSMOM30_DD"].iloc[1], 0.331)

    def test_experiment_has_four_fixed_scenarios(self):
        rng = np.random.default_rng(991)
        index = pd.date_range("2016-09-12", periods=1000, freq="B", tz="UTC")
        returns = pd.DataFrame({
            "SG_TREND_INDICATOR": rng.normal(0.0002, 0.007, len(index)),
            "VXTH": rng.normal(0.0003, 0.009, len(index)),
            "BTC_TSMOM30_DD": rng.normal(0.0008, 0.024, len(index)),
        }, index=index)
        summary, artifacts = report.run_experiment(returns, repetitions=99)
        expected = {"1x", "2x", "3x", "RISK_OVERLAY"}
        self.assertEqual(set(summary["scenario"]), expected)
        self.assertEqual(set(artifacts), expected)
        self.assertIn("passed", summary)


if __name__ == "__main__":
    unittest.main()
