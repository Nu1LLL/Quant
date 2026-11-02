import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER))

from alphas.base import make_signal
from research import ResearchRegistry, ResearchRunConfig, ResearchRunner, alpha_to_exposure


def make_data(rows=500):
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.006, rows)
    close = 100 * np.cumprod(1 + returns)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2020-01-01", periods=rows, freq="4h", tz="UTC"),
            "open": open_, "high": np.maximum(open_, close) + 1,
            "low": np.minimum(open_, close) - 1, "close": close,
            "volume": 1000.0,
        }
    )


class ResearchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.data = make_data()
        raw = pd.Series(np.sin(np.arange(len(self.data)) / 20), index=self.data.index)
        self.alpha = make_signal("test", raw, raw, "long", 20)

    def test_alpha_adapter_and_continuous_engine(self):
        exposure = alpha_to_exposure(self.alpha)
        self.assertTrue(((exposure >= 0) & (exposure <= 1)).all())
        result = ResearchRunner().run_exposure(
            self.data, exposure, ResearchRunConfig(initial_capital=10_000)
        )
        self.assertEqual(result["engine"], "continuous_exposure")
        self.assertIn("sharpe_ratio", result["metrics"])
        self.assertEqual(len(result["equity_curve"]), len(self.data))

    def test_registered_run_records_result_and_report_manifest(self):
        registry = ResearchRegistry(":memory:")
        registry.preregister(
            "trial", 100, 200,
            {
                "signal_versions": ["test:1"], "dataset_versions": ["fixture:1"],
                "holding_seconds": [14400], "cost_model_version": "spot-v1",
                "split_policy": "forward", "acceptance_rules": {"sharpe": 0},
            },
        )
        runner = ResearchRunner(registry)
        result = runner.run_exposure(
            self.data, alpha_to_exposure(self.alpha), experiment_id="trial"
        )
        with tempfile.TemporaryDirectory() as folder:
            manifest = runner.write_report(
                result, folder, "trial", provenance={"dataset": "fixture:1"}
            )
            self.assertEqual(manifest["provenance"]["dataset"], "fixture:1")
            loaded = json.loads(Path(folder, "trial_manifest.json").read_text())
            self.assertEqual(loaded["engine"], "continuous_exposure")
        registry.close()

    def test_unregistered_experiment_is_rejected(self):
        registry = ResearchRegistry(":memory:")
        with self.assertRaises(ValueError):
            ResearchRunner(registry).run_exposure(
                self.data, alpha_to_exposure(self.alpha), experiment_id="missing"
            )
        registry.close()


if __name__ == "__main__":
    unittest.main()
