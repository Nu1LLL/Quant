"""Synthetic-only tests for the one-shot spot-flow research runner."""

import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spot_Multi_Strategy.research import run_spot_flow as runner  # noqa: E402
from Spot_Multi_Strategy.research.spot_flow_archive import ArchiveIntegrityError  # noqa: E402


def _metrics(*, cagr=.5, sharpe=2.0, mdd=-.1, pf=2.0, net=.2, complete=True):
    return {"cagr": cagr, "sharpe": sharpe,
            "max_drawdown_intrabar": mdd,
            "profit_factor_daily": pf, "net_return": net,
            "complete_calendar_year": complete}


def _diagnostics():
    annual = {str(year): _metrics() for year in range(2022, 2026)}
    annual["2026"] = _metrics(complete=False)
    monte_carlo = {"joint_three_gate_frequency": .5,
                   "joint_four_gate_frequency": .4,
                   "positive_cagr_frequency": .99}
    return {"full": _metrics(), "holdout": _metrics(), "annual": annual,
            "regimes": {label: {"days": 200} for label in ("bull", "bear", "sideways")},
            "holdout_positive_month_fraction": .9,
            "holdout_worst_month": -.02,
            "monte_carlo_30d_10000": monte_carlo,
            "monte_carlo_v2_21d_10000": monte_carlo}


class SpotFlowRunnerTests(TestCase):
    def test_strong_historical_numbers_still_fail_independent_oos_gate(self):
        gate = runner._gate(_diagnostics())
        self.assertTrue(gate["historical_numeric_diagnostics_pass"])
        self.assertFalse(gate["full_unified_gate_passed"])
        self.assertFalse(gate["evidence"]["independent_oos_at_least_5y"])
        self.assertEqual(gate["oos_complete_years"], ["2024", "2025"])

    def test_numeric_failure_is_reported_not_rescued(self):
        diagnostics = _diagnostics()
        diagnostics["holdout"]["sharpe"] = .8
        gate = runner._gate(diagnostics)
        self.assertFalse(gate["historical_holdout_numeric"]["sharpe_above_1_5"])
        self.assertFalse(gate["historical_numeric_diagnostics_pass"])
        self.assertFalse(gate["full_unified_gate_passed"])

    def test_compact_manifest_keeps_digest_not_raw_archive_entries(self):
        manifest = {"sources": {"spot": "official", "futures": "official"},
                    "symbols": ["BTCUSDT"], "start_month": "2021-01",
                    "end_month": "2026-09", "archive_count": 1,
                    "archives": {"spot/BTCUSDT/2021-01": {"sha256": "abc"}},
                    "coverage": {"BTCUSDT": {"spot": {"coverage": 1.0}}}}
        compact = runner._compact_manifest(manifest)
        self.assertNotIn("archives", compact)
        self.assertEqual(len(compact["full_canonical_manifest_sha256"]), 64)
        json.dumps(compact, allow_nan=False)

    def test_archive_failure_seals_without_backtest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (mock.patch.object(runner, "REPORT_DIR", Path(temp_dir)),
                  mock.patch.object(runner, "_git_output", side_effect=["", "abc123"]),
                  mock.patch.object(runner, "load_and_audit",
                                    side_effect=ArchiveIntegrityError("bad source")),
                  mock.patch.object(runner, "run_backtest") as backtest,
                  mock.patch.object(sys, "argv", ["run_spot_flow.py"])):
                self.assertEqual(runner.main(), 2)
                backtest.assert_not_called()
                result = json.loads((Path(temp_dir) / "DATA_FAILURE.json").read_text())
                self.assertFalse(result["full_unified_gate_passed"])
                self.assertFalse((Path(temp_dir) / "RESULT.json").exists())

    def test_existing_outcome_refuses_rerun(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "RESULT.json").write_text("{}", encoding="utf-8")
            with (mock.patch.object(runner, "REPORT_DIR", Path(temp_dir)),
                  mock.patch.object(runner, "load_and_audit") as download,
                  mock.patch.object(sys, "argv", ["run_spot_flow.py"])):
                with self.assertRaises(SystemExit):
                    runner.main()
                download.assert_not_called()
