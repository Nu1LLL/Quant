"""Synthetic-only FNG one-shot orchestration tests; no live source is read."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from research import run_fng_crypto as runner


SYMBOLS = ("BTCUSDT", "ETHUSDT", "DOGEUSDT", "LTCUSDT", "BCHUSDT")


def _fng():
    return SimpleNamespace(
        observations=("synthetic-unread-daily-observation",),
        manifest={"url": "synthetic://fng", "coverage_days": 1821},
    )


def _signal():
    start = pd.Timestamp("2020-12-30", tz="UTC")
    return tuple({
        "day": (start + pd.Timedelta(days=7 * index)).date().isoformat(),
        "value": 25 if index % 2 else 75,
        "direction": 1 if index % 2 else -1,
    } for index in range(runner.WEEK_COUNT))


def _crypto():
    return SimpleNamespace(
        trade_bars={symbol: object() for symbol in SYMBOLS},
        mark_bars={symbol: object() for symbol in SYMBOLS},
        funding_events={symbol: object() for symbol in SYMBOLS},
        manifest={"archive_count": 951, "source": "synthetic://binance"},
    )


def _ledger(final: float):
    return SimpleNamespace(
        final_equity=final, fee_usdt=10.0, slippage_usdt=20.0,
        funding_usdt=3.0, turnover_usdt=10_000.0,
        conservative_max_drawdown=-0.10,
        trades=(object(),), records=(SimpleNamespace(exposed=True),),
    )


def _diagnostics() -> dict:
    annual = {year: -0.05 for year in range(2021, 2026)}
    return {
        "numerical_gate": True,
        "strategy_execution": {"conservative_max_drawdown_by_year": annual},
        "benchmark_execution": {
            "conservative_max_drawdown_by_year": annual.copy(),
        },
    }


@contextmanager
def _ready(directory: Path):
    with patch.object(runner, "REPORT_DIR", directory), patch.object(
        runner.frozen, "_ancestor", return_value=True,
    ), patch.object(
        runner.frozen, "_committed_and_pushed", return_value=True,
    ), patch.object(
        runner.frozen, "_current_commit", return_value="synthetic-code-commit",
    ):
        yield


@contextmanager
def _valid_sources():
    with patch.object(runner, "load_fng", return_value=_fng()), patch.object(
        runner, "adapt_fng_observations", return_value=("adapted",),
    ), patch.object(
        runner, "signal_fingerprint", return_value=_signal(),
    ), patch.object(
        runner, "load_crypto", return_value=_crypto(),
    ):
        yield


class TestRunFngCrypto(unittest.TestCase):
    def test_repository_guards_precede_source_access(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(runner, "REPORT_DIR", directory), patch.object(
                runner.frozen, "_ancestor", return_value=False,
            ), patch.object(runner, "load_fng") as fetch:
                with self.assertRaisesRegex(RuntimeError, "preregistration"):
                    runner.run()
                fetch.assert_not_called()
            with patch.object(runner, "REPORT_DIR", directory), patch.object(
                runner.frozen, "_ancestor", return_value=True,
            ), patch.object(
                runner.frozen, "_committed_and_pushed", return_value=False,
            ), patch.object(runner, "load_fng") as fetch:
                with self.assertRaisesRegex(RuntimeError, "committed, pushed"):
                    runner.run()
                fetch.assert_not_called()
            with patch.object(runner, "REPORT_DIR", directory), patch.object(
                runner.frozen, "_ancestor", return_value=True,
            ), patch.object(
                runner.frozen, "_committed_and_pushed", return_value=True,
            ), patch.object(
                runner.frozen, "_current_commit",
                return_value=runner.PREREG_COMMIT + "0" * 33,
            ), patch.object(runner, "load_fng") as fetch:
                with self.assertRaisesRegex(RuntimeError, "separately committed"):
                    runner.run()
                fetch.assert_not_called()

    def test_existing_terminal_and_held_lock_prevent_source_access(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)):
            directory = Path(temporary)
            (directory / "RESULT.json").write_text("{}", encoding="utf-8")
            with patch.object(runner, "load_fng") as fetch:
                with self.assertRaisesRegex(FileExistsError, "already sealed"):
                    runner.run()
                fetch.assert_not_called()
        with TemporaryDirectory() as temporary, _ready(Path(temporary)):
            directory = Path(temporary)
            lock = directory / runner.LOCK_NAME
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            with patch.object(runner, "load_fng") as fetch:
                with self.assertRaisesRegex(FileExistsError, "lock is held"):
                    runner.run()
                fetch.assert_not_called()
            self.assertTrue(lock.exists())
            self.assertFalse((directory / "CODE_FAILURE.json").exists())

    def test_competing_terminal_does_not_become_code_failure(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)):
            directory = Path(temporary)
            real_open = os.open

            def competing_terminal(path, flags, mode=0o777):
                descriptor = real_open(path, flags, mode)
                (directory / "RESULT.json").write_text("{}", encoding="utf-8")
                return descriptor

            with patch.object(runner.os, "open", side_effect=competing_terminal), patch.object(
                runner, "load_fng",
            ) as fetch:
                with self.assertRaisesRegex(FileExistsError, "sealed by another process"):
                    runner.run()
                fetch.assert_not_called()
            self.assertFalse((directory / runner.LOCK_NAME).exists())
            self.assertFalse((directory / "CODE_FAILURE.json").exists())

    def test_fng_data_failure_stops_before_market_read(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fng", side_effect=runner.FNGIntegrityError("missing day"),
        ), patch.object(runner, "load_crypto") as market:
            path = runner.run()
            market.assert_not_called()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "DATA_FAILURE.json")
            self.assertFalse(result["pnl_computed"])
            self.assertFalse(result["pnl_published"])
            self.assertFalse(result["all_gate_passed"])
            self.assertFalse(result["independent_oos"])
            self.assertEqual(result["prereg_commit"], runner.PREREG_COMMIT)
            self.assertEqual(result["source_attribution"]["alternative_me_fng"]["credit"],
                             "Alternative.me Crypto Fear & Greed Index")

    def test_adapter_failure_stops_before_market_read(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fng", return_value=_fng(),
        ), patch.object(
            runner, "adapt_fng_observations",
            side_effect=runner.FNGIntegrityError("bad value"),
        ), patch.object(runner, "load_crypto") as market:
            path = runner.run()
            market.assert_not_called()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "DATA_FAILURE.json")
            self.assertEqual(result["sources"]["alternative_me_fng"]["coverage_days"], 1821)
            self.assertNotIn("signal", result)

    def test_market_failure_keeps_signal_and_fng_manifest(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), _valid_sources(), patch.object(
            runner, "load_crypto", side_effect=runner.ArchiveIntegrityError("bad bar"),
        ), patch.object(runner, "run_backtest") as simulate:
            path = runner.run()
            simulate.assert_not_called()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "DATA_FAILURE.json")
            self.assertEqual(set(result["sources"]), {"alternative_me_fng"})
            self.assertEqual(len(result["signal"]), 261)
            self.assertFalse(result["pnl_computed"])

    def test_simulation_data_error_suppresses_pnl(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), _valid_sources(), patch.object(
            runner, "run_backtest", side_effect=runner.SimulationDataError("bad grid"),
        ):
            path = runner.run()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "DATA_FAILURE.json")
            self.assertFalse(result["pnl_published"])
            self.assertEqual(set(result["sources"]), {"alternative_me_fng", "binance_usdm"})

    def test_metrics_data_error_is_sealed_as_data_failure(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), _valid_sources(), patch.object(
            runner, "run_backtest", side_effect=[_ledger(12_000.0)] * 3,
        ), patch.object(
            runner, "evaluate", side_effect=runner.MetricsDataError("incomplete regime grid"),
        ):
            path = runner.run()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "DATA_FAILURE.json")
            self.assertTrue(result["pnl_computed"])
            self.assertFalse(result["pnl_published"])
            self.assertEqual(result["error_type"], "MetricsDataError")
            self.assertFalse((Path(temporary) / "CODE_FAILURE.json").exists())

    def test_primary_risk_failure_is_sealed_without_full_metrics(self):
        breach = runner.MarginRuinError(
            pd.Timestamp("2022-01-02", tz="UTC"), 100.0, -20.0,
            "synthetic ruin",
        )
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), _valid_sources(), patch.object(
            runner, "run_backtest", side_effect=breach,
        ), patch.object(runner, "evaluate") as metrics:
            path = runner.run()
            metrics.assert_not_called()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "RESULT.json")
            self.assertEqual(result["status"], "sealed-risk-failure")
            self.assertEqual(len(result["signal"]), 261)
            self.assertFalse(result["full_sample_metrics_computed"])
            self.assertFalse(result["all_gate_passed"])

    def test_result_uses_three_frozen_scenarios_and_strict_json(self):
        primary, stress, control = _ledger(12_000.0), _ledger(11_000.0), _ledger(9_000.0)
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), _valid_sources(), patch.object(
            runner, "run_backtest", side_effect=[primary, stress, control],
        ) as simulate, patch.object(
            runner, "evaluate", return_value=_diagnostics(),
        ) as metrics:
            path = runner.run()
            self.assertEqual(path.name, "RESULT.json")
            self.assertEqual(simulate.call_count, 3)
            self.assertEqual([call.args[0] for call in simulate.call_args_list],
                             [("adapted",)] * 3)
            self.assertEqual([call.kwargs for call in simulate.call_args_list], [
                {"slippage_rate": 0.001},
                {"slippage_rate": 0.0025},
                {"slippage_rate": 0.001, "benchmark": True},
            ])
            metrics.assert_called_once_with(primary, control, simulate.call_args_list[0].args[1]["BTCUSDT"])
            result = json.loads(path.read_text())
            self.assertEqual(len(result["signal"]), 261)
            self.assertEqual(result["account"]["final_usdt"], 12_000.0)
            self.assertEqual(result["account"]["funding_usdt"], 3.0)
            self.assertEqual(result["account"]["stress_25bp_funding_usdt"], 3.0)
            self.assertEqual(result["account"]["equal_weight_benchmark_funding_usdt"], 3.0)
            self.assertTrue(result["diagnostics"]["numerical_and_stress_gate"])
            self.assertFalse(result["diagnostics"]["independent_oos"])
            self.assertFalse(result["all_gate_passed"])
            self.assertEqual(list(result["diagnostics"]["strategy_execution"][
                "conservative_max_drawdown_by_year"
            ]), [str(year) for year in range(2021, 2026)])
            self.assertEqual(list(result["diagnostics"]["benchmark_execution"][
                "conservative_max_drawdown_by_year"
            ]), [str(year) for year in range(2021, 2026)])
            self.assertFalse((Path(temporary) / runner.LOCK_NAME).exists())

    def test_stress_and_control_risk_are_reported_not_retried(self):
        breach = runner.MarginRuinError(
            pd.Timestamp("2022-01-02", tz="UTC"), 100.0, -20.0,
            "synthetic risk",
        )
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), _valid_sources(), patch.object(
            runner, "run_backtest", side_effect=[_ledger(12_000.0), breach, breach],
        ) as simulate, patch.object(
            runner, "evaluate", return_value=_diagnostics(),
        ) as metrics:
            path = runner.run()
            self.assertEqual(simulate.call_count, 3)
            metrics.assert_called_once()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "RESULT.json")
            self.assertFalse(result["diagnostics"]["numerical_and_stress_gate"])
            self.assertFalse(result["diagnostics"]["benchmark_path_complete"])
            self.assertIn("synthetic risk", result["account"]["stress_25bp_risk_failure"])
            self.assertIn("synthetic risk", result["account"]["equal_weight_benchmark_risk_failure"])

    def test_unexpected_error_and_bad_report_tree_seal_code_failure(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fng", side_effect=RuntimeError("synthetic parser bug"),
        ):
            path = runner.run()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "CODE_FAILURE.json")
            self.assertFalse(result["pnl_published"])
            self.assertFalse(result["all_gate_passed"])

        bad = _diagnostics()
        bad["unexpected_object"] = {2021: 0.0}
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), _valid_sources(), patch.object(
            runner, "run_backtest", side_effect=[_ledger(12_000.0)] * 3,
        ), patch.object(runner, "evaluate", return_value=bad):
            path = runner.run()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "CODE_FAILURE.json")
            self.assertIn("keys must be strings", result["reason"])
            self.assertFalse((Path(temporary) / "RESULT.json").exists())

    def test_crash_leaves_lock_for_manual_audit(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "_execute_once", side_effect=KeyboardInterrupt,
        ), patch.object(runner, "load_fng") as fetch:
            with self.assertRaises(KeyboardInterrupt):
                runner.run()
            self.assertTrue((Path(temporary) / runner.LOCK_NAME).exists())
            with self.assertRaisesRegex(FileExistsError, "lock is held"):
                runner.run()
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
