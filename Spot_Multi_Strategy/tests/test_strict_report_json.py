"""Standalone synthetic serializer contract tests (stdlib unittest)."""

from datetime import date
import json
import unittest

import numpy as np

from research.strict_report_json import dumps


class StrictReportJSONTests(unittest.TestCase):
    def test_nested_annual_keys_and_numpy_values_round_trip(self):
        payload = {"diagnostics": {"strategy_execution": {
            "conservative_max_drawdown_by_year": {
                2021: np.float64(-0.12), 2022: np.float64(-0.18),
            },
            "terminal_flat": np.bool_(True),
        }}, "sealed_on": date(2026, 11, 8)}
        restored = json.loads(dumps(payload))
        self.assertEqual(restored["diagnostics"]["strategy_execution"]
                         ["conservative_max_drawdown_by_year"],
                         {"2021": -0.12, "2022": -0.18})
        self.assertTrue(restored["diagnostics"]["strategy_execution"]["terminal_flat"])
        self.assertEqual(restored["sealed_on"], "2026-11-08")

    def test_rejects_collision_and_nonfinite_value(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            dumps({"2021": "a", 2021: "b"})
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            dumps({"metric": float("nan")})


if __name__ == "__main__":
    unittest.main()
