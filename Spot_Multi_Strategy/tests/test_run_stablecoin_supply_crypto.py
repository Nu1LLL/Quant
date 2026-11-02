"""Synthetic-only one-shot orchestration tests; never fetch source observations."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from research import run_stablecoin_supply_crypto as runner


def _supply():
    return SimpleNamespace(
        observations=(object(),), manifest={"rows_per_token": 1891},
    )


def _crypto():
    return SimpleNamespace(
        trade_bars={"BTCUSDT": object()}, mark_bars={"BTCUSDT": object()},
        funding_events={"BTCUSDT": object()}, manifest={"archive_count": 951},
    )


def _ledger(final: float):
    return SimpleNamespace(
        final_equity=final, fee_usdt=10.0, slippage_usdt=10.0,
        funding_usdt=1.0, turnover_usdt=10_000.0,
        conservative_max_drawdown=-0.10,
        trades=(object(),), records=(SimpleNamespace(exposed=True),),
    )


def _ready(monkeypatch, directory):
    monkeypatch.setattr(runner, "REPORT_DIR", directory)
    return (
        patch.object(runner.frozen, "_ancestor", return_value=True),
        patch.object(runner.frozen, "_committed_and_pushed", return_value=True),
        patch.object(runner.frozen, "_current_commit", return_value="synthetic-code"),
    )


def test_prereg_separate_pushed_code_and_clean_tree_required_before_sources(
        tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    with patch.object(runner.frozen, "_ancestor", return_value=False), patch.object(
        runner, "load_supply",
    ) as fetch:
        with pytest.raises(RuntimeError, match="preregistration"):
            runner.run()
        fetch.assert_not_called()
    with patch.object(runner.frozen, "_ancestor", return_value=True), patch.object(
        runner.frozen, "_committed_and_pushed", return_value=False,
    ), patch.object(runner, "load_supply") as fetch:
        with pytest.raises(RuntimeError, match="committed, pushed"):
            runner.run()
        fetch.assert_not_called()
    with patch.object(runner.frozen, "_ancestor", return_value=True), patch.object(
        runner.frozen, "_committed_and_pushed", return_value=True,
    ), patch.object(
        runner.frozen, "_current_commit", return_value=runner.PREREG_COMMIT + "0" * 33,
    ), patch.object(runner, "load_supply") as fetch:
        with pytest.raises(RuntimeError, match="separately committed"):
            runner.run()
        fetch.assert_not_called()


def test_source_failure_is_data_failure_and_prevents_market_read(tmp_path, monkeypatch):
    guards = _ready(monkeypatch, tmp_path)
    with guards[0], guards[1], guards[2], patch.object(
        runner, "load_supply", side_effect=runner.StablecoinDataError("missing date"),
    ), patch.object(runner, "load_crypto") as crypto, patch.object(
        runner, "run_backtest",
    ) as simulate:
        path = runner.run()
        crypto.assert_not_called()
        simulate.assert_not_called()
    result = json.loads(path.read_text())
    assert path.name == "DATA_FAILURE.json"
    assert result["pnl_computed"] is False
    assert result["pnl_published"] is False
    assert result["all_gate_passed"] is False
    assert result["prereg_commit"] == "09682e7"
    assert result["evidence_grade"] == "retrospective_backfilled_supply_proxy"
    assert result["independent_oos"] is False
    assert result["signal_publication_proxy"]["actual_first_publication_time_verified"] is False
    assert "fed_h41" not in result.get("sources", {})
    with patch.object(runner.frozen, "_ancestor", return_value=True):
        with pytest.raises(FileExistsError, match="already sealed"):
            runner.run()


def test_binance_failure_retains_only_supply_manifest(tmp_path, monkeypatch):
    guards = _ready(monkeypatch, tmp_path)
    with guards[0], guards[1], guards[2], patch.object(
        runner, "load_supply", return_value=_supply(),
    ), patch.object(
        runner, "load_crypto", side_effect=runner.ArchiveIntegrityError("bad bar"),
    ), patch.object(runner, "run_backtest") as simulate:
        path = runner.run()
        simulate.assert_not_called()
    result = json.loads(path.read_text())
    assert result["sources"] == {
        "defillama_stablecoin_supply": {"rows_per_token": 1891},
    }
    assert result["pnl_computed"] is False


def test_simulation_data_failure_is_not_published_as_pnl(tmp_path, monkeypatch):
    guards = _ready(monkeypatch, tmp_path)
    with guards[0], guards[1], guards[2], patch.object(
        runner, "load_supply", return_value=_supply(),
    ), patch.object(runner, "load_crypto", return_value=_crypto()), patch.object(
        runner, "run_backtest", side_effect=runner.SimulationDataError("bad grid"),
    ):
        path = runner.run()
    result = json.loads(path.read_text())
    assert path.name == "DATA_FAILURE.json"
    assert result["pnl_published"] is False
    assert set(result["sources"]) == {
        "defillama_stablecoin_supply", "binance_usdm",
    }


def test_margin_and_unexpected_code_errors_seal_distinctly(tmp_path, monkeypatch):
    risk_dir = tmp_path / "risk"
    guards = _ready(monkeypatch, risk_dir)
    breach = runner.MarginRuinError(
        pd.Timestamp("2022-01-02", tz="UTC"), 100.0, -20.0,
        "synthetic ruin",
    )
    with guards[0], guards[1], guards[2], patch.object(
        runner, "load_supply", return_value=_supply(),
    ), patch.object(runner, "load_crypto", return_value=_crypto()), patch.object(
        runner, "run_backtest", side_effect=breach,
    ):
        risk_path = runner.run()
    risk = json.loads(risk_path.read_text())
    assert risk_path.name == "RESULT.json"
    assert risk["status"] == "sealed-risk-failure"
    assert risk["all_gate_passed"] is False

    code_dir = tmp_path / "code"
    guards = _ready(monkeypatch, code_dir)
    with guards[0], guards[1], guards[2], patch.object(
        runner, "load_supply", side_effect=RuntimeError("synthetic parser bug"),
    ):
        code_path = runner.run()
    code = json.loads(code_path.read_text())
    assert code_path.name == "CODE_FAILURE.json"
    assert code["pnl_published"] is False
    assert code["all_gate_passed"] is False


def test_result_uses_three_registered_scenarios_and_correct_source(tmp_path, monkeypatch):
    guards = _ready(monkeypatch, tmp_path)
    primary, stress, control = (
        _ledger(12_000.0), _ledger(11_000.0), _ledger(9_000.0),
    )
    supply, crypto = _supply(), _crypto()
    with guards[0], guards[1], guards[2], patch.object(
        runner, "load_supply", return_value=supply,
    ), patch.object(runner, "load_crypto", return_value=crypto), patch.object(
        runner, "run_backtest", side_effect=[primary, stress, control],
    ) as simulate, patch.object(
        runner, "evaluate", return_value={"full_gate": True, "evidence_grade": "fed"},
    ) as metrics:
        path = runner.run()
    assert simulate.call_count == 3
    assert [call.kwargs for call in simulate.call_args_list] == [
        {"slippage_rate": 0.001},
        {"slippage_rate": 0.0025},
        {"slippage_rate": 0.001, "benchmark": True},
    ]
    metrics.assert_called_once_with(primary, control, crypto.trade_bars["BTCUSDT"])
    result = json.loads(path.read_text())
    assert path.name == "RESULT.json"
    assert result["sources"]["defillama_stablecoin_supply"] == supply.manifest
    assert "fed_h41" not in result["sources"]
    assert result["sources"]["binance_usdm"] == crypto.manifest
    assert result["diagnostics"]["evidence_grade"] == runner.EVIDENCE_GRADE
    assert result["diagnostics"]["full_gate"] is False
    assert result["all_gate_passed"] is False
    assert result["account"]["final_usdt"] == 12_000.0


def test_stress_and_control_risk_failure_are_diagnostics_only(tmp_path, monkeypatch):
    guards = _ready(monkeypatch, tmp_path)
    primary = _ledger(12_000.0)
    breach = runner.MarginRuinError(
        pd.Timestamp("2022-01-02", tz="UTC"), 100.0, -20.0, "synthetic",
    )
    with guards[0], guards[1], guards[2], patch.object(
        runner, "load_supply", return_value=_supply(),
    ), patch.object(runner, "load_crypto", return_value=_crypto()), patch.object(
        runner, "run_backtest", side_effect=[primary, breach, breach],
    ), patch.object(runner, "evaluate", return_value={}) as metrics:
        path = runner.run()
    result = json.loads(path.read_text())
    assert path.name == "RESULT.json"
    assert result["account"]["stress_25bp_risk_failure"]
    assert result["account"]["equal_weight_benchmark_risk_failure"]
    metrics.assert_called_once()
