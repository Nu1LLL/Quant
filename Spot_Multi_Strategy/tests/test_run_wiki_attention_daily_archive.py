"""Fixture-only orchestration tests; no historical values or APIs are read."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from research import run_wiki_attention_daily_archive as runner


def _bundle():
    return SimpleNamespace(
        views={"BTCUSDT": object()}, trade_bars={"BTCUSDT": object()},
        mark_bars={"BTCUSDT": object()}, funding_events={"BTCUSDT": object()},
        manifest={"archive_count": 0, "daily_replacements": []},
    )


def _ledger(final: float):
    return SimpleNamespace(
        final_equity=final, fee_usdt=10.0, slippage_usdt=10.0,
        funding_usdt=1.0, turnover_usdt=10_000.0,
        conservative_max_drawdown=-0.10,
        trades=(object(),), records=(SimpleNamespace(exposed=True),),
    )


def test_prereg_ancestor_required_before_source_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    with patch.object(runner, "_ancestor", return_value=False), patch.object(
        runner, "load_and_audit"
    ) as fetch:
        with pytest.raises(RuntimeError, match="preregistration"):
            runner.run()
        fetch.assert_not_called()


def test_dirty_or_unpushed_code_is_blocked_before_source_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=False
    ), patch.object(runner, "load_and_audit") as fetch:
        with pytest.raises(RuntimeError, match="committed, pushed"):
            runner.run()
        fetch.assert_not_called()


def test_data_failure_is_sealed_without_pnl(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=True
    ), patch.object(
        runner, "_current_commit", return_value="synthetic-code"
    ), patch.object(runner, "load_and_audit", side_effect=runner.ArchiveIntegrityError(
        "missing daily ZIP"
    )), patch.object(runner, "run_backtest") as simulate:
        path = runner.run()
        simulate.assert_not_called()
    result = json.loads(path.read_text())
    assert path.name == "DATA_FAILURE.json"
    assert result["pnl_computed"] is False
    assert result["independent_oos"] is False
    assert result["prereg_commit"] == "2708aa3"
    with patch.object(runner, "_ancestor", return_value=True):
        with pytest.raises(FileExistsError, match="already sealed"):
            runner.run()


def test_risk_failure_seals_without_extrapolated_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    breach = runner.MarginRuinError(pd.Timestamp("2022-01-02", tz="UTC"),
                                    100.0, -20.0, "synthetic ruin")
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=True
    ), patch.object(
        runner, "_current_commit", return_value="synthetic-code"
    ), patch.object(runner, "load_and_audit", return_value=_bundle()), patch.object(
        runner, "run_backtest", side_effect=breach
    ), patch.object(runner, "evaluate") as metrics:
        path = runner.run()
        metrics.assert_not_called()
    result = json.loads(path.read_text())
    assert result["status"] == "sealed-risk-failure"
    assert result["full_sample_metrics_computed"] is False
    assert result["all_gate_passed"] is False


def test_unexpected_code_failure_seals_one_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=True
    ), patch.object(runner, "_current_commit", return_value="synthetic-code"), patch.object(
        runner, "load_and_audit", side_effect=RuntimeError("synthetic parser bug")
    ):
        path = runner.run()
    result = json.loads(path.read_text())
    assert path.name == "CODE_FAILURE.json"
    assert result["status"] == "code-failure"
    assert result["all_gate_passed"] is False
    with patch.object(runner, "_ancestor", return_value=True):
        with pytest.raises(FileExistsError, match="already sealed"):
            runner.run()


def test_fixed_scenarios_and_separate_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    primary, stress, benchmark = _ledger(11_000.0), _ledger(10_500.0), _ledger(9_000.0)
    bundle = _bundle()
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=True
    ), patch.object(
        runner, "_current_commit", return_value="synthetic-code"
    ), patch.object(runner, "load_and_audit", return_value=bundle), patch.object(
        runner, "run_backtest", side_effect=[primary, stress, benchmark]
    ) as simulate, patch.object(runner, "evaluate", return_value={
        "independent_oos": False, "full_gate": False,
    }) as metrics:
        path = runner.run()
        assert simulate.call_count == 3
        assert simulate.call_args_list[0].kwargs == {"slippage_rate": 0.001}
        assert simulate.call_args_list[1].kwargs == {"slippage_rate": 0.0025}
        assert simulate.call_args_list[2].kwargs == {
            "slippage_rate": 0.001, "benchmark": True,
        }
        metrics.assert_called_once_with(primary, benchmark, bundle.trade_bars["BTCUSDT"])
    result = json.loads(path.read_text())
    assert result["status"] == "sealed-result"
    assert result["sources"]["daily_replacements"] == []
    assert result["account"]["final_usdt"] == 11_000.0
    assert result["account"]["stress_25bp_final_usdt"] == 10_500.0
    assert result["account"]["equal_weight_benchmark_final_usdt"] == 9_000.0
