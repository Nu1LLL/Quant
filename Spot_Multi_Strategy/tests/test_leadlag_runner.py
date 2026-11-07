from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from research import leadlag_runner
from research.leadlag_sim import LeadLagStrategyFailure


class LeadLagRunnerTests(unittest.TestCase):
    def test_failed_stress_is_sealed_once_without_partial_metrics(self):
        with TemporaryDirectory() as directory:
            report_path = Path(directory) / "sealed.json"
            with (patch.object(leadlag_runner, "REPORT_PATH", report_path),
                  patch.object(leadlag_runner, "run_one_shot",
                               side_effect=LeadLagStrategyFailure("stress: nonpositive equity"))):
                leadlag_runner.main()
                result = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(result["status"], "sealed_failed_run")
                self.assertFalse(result["gate"]["full"])
                self.assertNotIn("metrics", result)
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    leadlag_runner.main()


if __name__ == "__main__":
    unittest.main()
