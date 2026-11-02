import sqlite3
import sys
import unittest
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))

from research.registry import ResearchRegistry


class ResearchRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ResearchRegistry(":memory:")
        self.registry.register_signal(
            "news", "1", "event", "news-expert", {"fields": ["surprise"]}
        )

    def tearDown(self):
        self.registry.close()

    def specification(self):
        return {
            "signal_versions": ["news:1"],
            "dataset_versions": ["archive-1"],
            "holding_seconds": [60, 300],
            "cost_model_version": "spot-v1",
            "split_policy": "forward",
            "acceptance_rules": {"net_lower_bound": 0},
        }

    def test_availability_time_blocks_late_observation(self):
        self.registry.observe(
            "news", "1", "equity", "TEST", 100, 200,
            "archive-1", {"surprise": 1},
        )
        self.assertEqual(self.registry.available(150), [])
        self.assertEqual(len(self.registry.available(200)), 1)

    def test_preregistration_and_single_final_result(self):
        self.registry.preregister("trial", 100, 200, self.specification())
        self.registry.record_result("trial", 300, {"sharpe": 1.0})
        with self.assertRaises(sqlite3.IntegrityError):
            self.registry.record_result("trial", 400, {"sharpe": 2.0})

    def test_evidence_tables_are_immutable(self):
        with self.assertRaises(sqlite3.IntegrityError):
            with self.registry.db:
                self.registry.db.execute("UPDATE signals SET owner='x'")

    def test_oos_must_follow_registration(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.registry.preregister("trial", 200, 100, self.specification())


if __name__ == "__main__":
    unittest.main()
