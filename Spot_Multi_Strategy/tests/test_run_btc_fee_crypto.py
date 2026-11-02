"""Synthetic-only one-shot orchestration tests; never fetch live sources."""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from research import run_btc_fee_crypto as runner


def _observations():
    start = pd.Timestamp("2020-10-28", tz="UTC")
    return tuple(
        SimpleNamespace(
            observed_date=start + pd.Timedelta(days=7 * index),
            release_date=start + pd.Timedelta(days=7 * index + 1),
            value=Decimal("2") if index in (4, 270) else Decimal("1"),
        )
        for index in range(runner.WEEK_COUNT)
    )


def _fee():
    return SimpleNamespace(
        observations=_observations(), manifest={"daily_count": 1918},
    )


def _crypto():
    return SimpleNamespace(
        trade_bars={"BTCUSDT": object()}, mark_bars={"BTCUSDT": object()},
        funding_events={"BTCUSDT": object()}, manifest={"archive_count": 951},
    )


def _ledger(final: float):
    return SimpleNamespace(
        final_equity=final, fee_usdt=10.0, slippage_usdt=20.0,
        funding_usdt=3.0, turnover_usdt=10_000.0,
        conservative_max_drawdown=-0.10,
        trades=(object(),), records=(SimpleNamespace(exposed=True),),
    )


@contextmanager
def _ready(directory: Path):
    with patch.object(runner, "REPORT_DIR", directory), patch.object(
        runner.frozen, "_ancestor", return_value=True,
    ), patch.object(
        runner.frozen, "_committed_and_pushed", return_value=True,
    ), patch.object(runner.frozen, "_current_commit", return_value="synthetic-code"):
        yield


class TestRunBtcFeeCrypto(unittest.TestCase):
    def test_report_json_preserves_numpy_numeric_types_and_rejects_nan(self):
        with TemporaryDirectory() as temporary, patch.object(
            runner, "REPORT_DIR", Path(temporary),
        ):
            path = runner._write_once("RESULT.json", {
                "counts": np.int64(7), "ratio": np.float32(0.5),
                "passed": np.bool_(True),
            })
            result = json.loads(path.read_text())
            self.assertEqual(result, {"counts": 7, "ratio": 0.5, "passed": True})
            self.assertIsInstance(result["counts"], int)
            self.assertIsInstance(result["ratio"], float)
        with TemporaryDirectory() as temporary, patch.object(
            runner, "REPORT_DIR", Path(temporary),
        ):
            for bad in (float("nan"), np.float32("nan"), np.float64("inf")):
                with self.subTest(bad=type(bad).__name__):
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        runner._write_once("RESULT.json", {"nested": [bad]})
                    self.assertFalse((runner.REPORT_DIR / "RESULT.json").exists())
            with self.assertRaisesRegex(TypeError, "unsupported"):
                runner._write_once("RESULT.json", {"numeric": Decimal("1.1")})
            self.assertFalse((runner.REPORT_DIR / "RESULT.json").exists())

    def test_held_one_shot_lock_fails_closed_before_signal_read(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            lock_path = directory / runner.LOCK_NAME
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            with _ready(directory), patch.object(runner, "load_fee") as fetch:
                with self.assertRaisesRegex(FileExistsError, "lock is held"):
                    runner.run()
                fetch.assert_not_called()
                self.assertTrue(lock_path.exists())
                self.assertFalse((directory / "CODE_FAILURE.json").exists())

    def test_terminal_appearing_after_lock_is_rechecked_without_source_read(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            real_open = os.open

            def competing_terminal(path, flags, mode=0o777):
                descriptor = real_open(path, flags, mode)
                (directory / "RESULT.json").write_text("{}", encoding="utf-8")
                return descriptor

            with _ready(directory), patch.object(runner.os, "open", side_effect=competing_terminal), patch.object(
                runner, "load_fee",
            ) as fetch:
                with self.assertRaisesRegex(FileExistsError, "sealed by another process"):
                    runner.run()
                fetch.assert_not_called()
                self.assertFalse((directory / "CODE_FAILURE.json").exists())
                self.assertFalse((directory / runner.LOCK_NAME).exists())

    def test_terminal_write_conflict_never_becomes_code_failure(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with _ready(directory), patch.object(
                runner, "_execute_once", side_effect=FileExistsError("simulated RESULT conflict"),
            ):
                with self.assertRaisesRegex(FileExistsError, "RESULT conflict"):
                    runner.run()
                self.assertFalse((directory / "CODE_FAILURE.json").exists())
                self.assertTrue((directory / runner.LOCK_NAME).exists())

    def test_crash_residual_lock_requires_manual_audit(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with _ready(directory), patch.object(
                runner, "_execute_once", side_effect=KeyboardInterrupt,
            ), patch.object(runner, "load_fee") as fetch:
                with self.assertRaises(KeyboardInterrupt):
                    runner.run()
                self.assertTrue((directory / runner.LOCK_NAME).exists())
                with self.assertRaisesRegex(FileExistsError, "lock is held"):
                    runner.run()
                fetch.assert_not_called()
                self.assertFalse((directory / "CODE_FAILURE.json").exists())

    def test_direction_hex_bit_order_and_adapter_long_cash_consistency(self):
        raw = _observations()
        adapted = runner.adapt_long_cash_observations(raw)
        signal = runner._direction_fingerprint(raw, adapted)
        self.assertEqual(signal["weekly_direction_count"], 271)
        self.assertEqual(signal["weekly_direction_warmup_zero_bits"], 4)
        self.assertEqual(signal["weekly_direction_padding_bits"], 1)
        self.assertEqual(len(signal["weekly_direction_bits_hex"]), 68)
        packed = bytes.fromhex(signal["weekly_direction_bits_hex"])
        bits = "".join(f"{byte:08b}" for byte in packed)
        self.assertEqual(bits[:271], "00001" + "0" * 265 + "1")
        self.assertEqual(bits[271], "0")
        self.assertEqual(signal["weekly_direction_long_count"], 2)
        for index in range(4, 271):
            expected = raw[index].value > raw[index - 4].value
            actual = adapted[index].value > adapted[index - 4].value
            self.assertEqual(actual, expected)
            self.assertGreaterEqual(adapted[index].value, adapted[index - 4].value)

    def test_tiny_decimal_increase_is_not_rounded_to_cash(self):
        raw = list(_observations())
        raw[4] = SimpleNamespace(
            observed_date=raw[4].observed_date,
            release_date=raw[4].release_date,
            value=Decimal("1.0000000000000000000000000001"),
        )
        adapted = runner.adapt_long_cash_observations(raw)
        signal = runner._direction_fingerprint(raw, adapted)
        self.assertTrue(signal["weekly_direction_bits_hex"].startswith("08"))
        self.assertEqual(adapted[4].value - adapted[0].value, 1)
        corrupt = list(adapted)
        corrupt[4] = SimpleNamespace(value=0)
        with self.assertRaisesRegex(runner.BtcFeeDataError, "disagrees"):
            runner._direction_fingerprint(raw, corrupt)

    def test_preflight_blocks_sources_until_separate_push(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(runner, "REPORT_DIR", directory), patch.object(
                runner.frozen, "_ancestor", return_value=False,
            ), patch.object(runner, "load_fee") as fetch:
                with self.assertRaisesRegex(RuntimeError, "preregistration"):
                    runner.run()
                fetch.assert_not_called()
            with patch.object(runner, "REPORT_DIR", directory), patch.object(
                runner.frozen, "_ancestor", return_value=True,
            ), patch.object(runner.frozen, "_committed_and_pushed", return_value=False), patch.object(
                runner, "load_fee",
            ) as fetch:
                with self.assertRaisesRegex(RuntimeError, "committed, pushed"):
                    runner.run()
                fetch.assert_not_called()
            with patch.object(runner, "REPORT_DIR", directory), patch.object(
                runner.frozen, "_ancestor", return_value=True,
            ), patch.object(runner.frozen, "_committed_and_pushed", return_value=True), patch.object(
                runner.frozen, "_current_commit", return_value=runner.PREREG_COMMIT + "0" * 33,
            ), patch.object(runner, "load_fee") as fetch:
                with self.assertRaisesRegex(RuntimeError, "separately committed"):
                    runner.run()
                fetch.assert_not_called()

    def test_fee_failure_seals_data_failure_before_market_read(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fee", side_effect=runner.BtcFeeDataError("missing date"),
        ), patch.object(runner, "load_crypto") as crypto, patch.object(
            runner, "run_backtest",
        ) as simulate:
            path = runner.run()
            crypto.assert_not_called()
            simulate.assert_not_called()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "DATA_FAILURE.json")
            self.assertIs(result["pnl_computed"], False)
            self.assertIs(result["pnl_published"], False)
            self.assertIs(result["all_gate_passed"], False)
            self.assertEqual(result["prereg_commit"], "70ee332")
            self.assertEqual(result["evidence_grade"], runner.EVIDENCE_GRADE)
            self.assertIs(result["independent_oos"], False)
            self.assertIs(result["full_gate"], False)
            self.assertEqual(result["source_attribution"]["coin_metrics"]["license"], "CC BY-NC 4.0")
            self.assertEqual(result["source_attribution"]["binance_vision"]["license"], "CC BY-NC-SA 4.0")
            self.assertIs(result["signal_publication_proxy"]["actual_first_publication_time_verified"], False)
            with self.assertRaisesRegex(FileExistsError, "already sealed"):
                runner.run()

    def test_adapter_failure_prevents_market_read(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fee", return_value=_fee(),
        ), patch.object(
            runner, "adapt_long_cash_observations",
            side_effect=runner.BtcFeeDataError("bad adapter input"),
        ), patch.object(runner, "load_crypto") as crypto:
            path = runner.run()
            crypto.assert_not_called()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "DATA_FAILURE.json")
            self.assertEqual(result["sources"], {"coinmetrics_btc_fee": {"daily_count": 1918}})
            self.assertIs(result["pnl_published"], False)

    def test_binance_failure_keeps_only_fee_manifest(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fee", return_value=_fee(),
        ), patch.object(
            runner, "load_crypto", side_effect=runner.ArchiveIntegrityError("bad bar"),
        ), patch.object(runner, "run_backtest") as simulate:
            path = runner.run()
            simulate.assert_not_called()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "DATA_FAILURE.json")
            self.assertEqual(result["sources"], {"coinmetrics_btc_fee": {"daily_count": 1918}})
            self.assertEqual(result["signal"]["weekly_direction_count"], 271)
            self.assertIs(result["pnl_computed"], False)

    def test_simulation_data_failure_does_not_publish_pnl(self):
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fee", return_value=_fee(),
        ), patch.object(runner, "load_crypto", return_value=_crypto()), patch.object(
            runner, "run_backtest", side_effect=runner.SimulationDataError("bad grid"),
        ):
            path = runner.run()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "DATA_FAILURE.json")
            self.assertIs(result["pnl_published"], False)
            self.assertEqual(set(result["sources"]), {"coinmetrics_btc_fee", "binance_usdm"})

    def test_margin_and_unexpected_code_errors_seal_distinctly(self):
        breach = runner.MarginRuinError(
            pd.Timestamp("2022-01-02", tz="UTC"), 100.0, -20.0,
            "synthetic ruin",
        )
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fee", return_value=_fee(),
        ), patch.object(runner, "load_crypto", return_value=_crypto()), patch.object(
            runner, "run_backtest", side_effect=breach,
        ):
            path = runner.run()
            risk = json.loads(path.read_text())
            self.assertEqual(path.name, "RESULT.json")
            self.assertEqual(risk["status"], "sealed-risk-failure")
            self.assertEqual(risk["signal"]["weekly_direction_count"], 271)
            self.assertIs(risk["all_gate_passed"], False)

        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fee", side_effect=RuntimeError("synthetic parser bug"),
        ):
            path = runner.run()
            code = json.loads(path.read_text())
            self.assertEqual(path.name, "CODE_FAILURE.json")
            self.assertIs(code["pnl_published"], False)
            self.assertIs(code["all_gate_passed"], False)

    def test_result_uses_adapted_three_scenarios_and_full_costs(self):
        primary, stress, control = _ledger(12_000.0), _ledger(11_000.0), _ledger(9_000.0)
        fee, crypto = _fee(), _crypto()
        adapted = runner.adapt_long_cash_observations(fee.observations)
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fee", return_value=fee,
        ), patch.object(
            runner, "adapt_long_cash_observations", return_value=adapted,
        ) as adapter, patch.object(
            runner, "load_crypto", return_value=crypto,
        ), patch.object(
            runner, "run_backtest", side_effect=[primary, stress, control],
        ) as simulate, patch.object(
            runner, "evaluate", return_value={"full_gate": True, "evidence_grade": "fed"},
        ) as metrics:
            path = runner.run()
            adapter.assert_called_once_with(fee.observations)
            self.assertEqual(simulate.call_count, 3)
            self.assertEqual([call.args[0] for call in simulate.call_args_list], [adapted] * 3)
            self.assertEqual([call.kwargs for call in simulate.call_args_list], [
                {"slippage_rate": 0.001},
                {"slippage_rate": 0.0025},
                {"slippage_rate": 0.001, "benchmark": True},
            ])
            metrics.assert_called_once_with(primary, control, crypto.trade_bars["BTCUSDT"])
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "RESULT.json")
            self.assertEqual(result["sources"]["coinmetrics_btc_fee"], fee.manifest)
            self.assertEqual(result["sources"]["binance_usdm"], crypto.manifest)
            self.assertTrue(result["signal"]["weekly_direction_bits_hex"].startswith("08"))
            self.assertEqual(result["diagnostics"]["evidence_grade"], runner.EVIDENCE_GRADE)
            self.assertIs(result["diagnostics"]["full_gate"], False)
            self.assertIs(result["all_gate_passed"], False)
            self.assertEqual(result["account"]["final_usdt"], 12_000.0)
            self.assertEqual(result["account"]["funding_usdt"], 3.0)
            self.assertEqual(result["account"]["equal_weight_benchmark_funding_usdt"], 3.0)
            self.assertFalse((Path(temporary) / runner.LOCK_NAME).exists())

    def test_stress_and_control_risk_failure_are_diagnostics_only(self):
        breach = runner.MarginRuinError(
            pd.Timestamp("2022-01-02", tz="UTC"), 100.0, -20.0, "synthetic",
        )
        with TemporaryDirectory() as temporary, _ready(Path(temporary)), patch.object(
            runner, "load_fee", return_value=_fee(),
        ), patch.object(runner, "load_crypto", return_value=_crypto()), patch.object(
            runner, "run_backtest", side_effect=[_ledger(12_000.0), breach, breach],
        ), patch.object(runner, "evaluate", return_value={}) as metrics:
            path = runner.run()
            result = json.loads(path.read_text())
            self.assertEqual(path.name, "RESULT.json")
            self.assertTrue(result["account"]["stress_25bp_risk_failure"])
            self.assertTrue(result["account"]["equal_weight_benchmark_risk_failure"])
            metrics.assert_called_once()


if __name__ == "__main__":
    unittest.main()
