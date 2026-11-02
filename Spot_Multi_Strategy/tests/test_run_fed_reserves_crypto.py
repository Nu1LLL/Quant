"""Fixture-only orchestration checks; no macro value or price source is read."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from research import run_fed_reserves_crypto as runner


def _fed():
    return SimpleNamespace(observations=(object(),), manifest={"release_count": 269})


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


def test_prereg_and_clean_pushed_code_are_required_before_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    with patch.object(runner, "_ancestor", return_value=False), patch.object(
        runner, "load_fed"
    ) as fetch:
        with pytest.raises(RuntimeError, match="preregistration"):
            runner.run()
        fetch.assert_not_called()
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=False
    ), patch.object(runner, "load_fed") as fetch:
        with pytest.raises(RuntimeError, match="committed, pushed"):
            runner.run()
        fetch.assert_not_called()


def test_fed_data_failure_prevents_crypto_fetch_and_pnl(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=True
    ), patch.object(runner, "_current_commit", return_value="synthetic-code"), patch.object(
        runner, "load_fed", side_effect=runner.VintageDataError("missing H.4.1 release")
    ), patch.object(runner, "load_crypto") as crypto, patch.object(
        runner, "run_backtest"
    ) as simulate:
        path = runner.run()
        crypto.assert_not_called()
        simulate.assert_not_called()
    result = json.loads(path.read_text())
    assert path.name == "DATA_FAILURE.json"
    assert result["pnl_computed"] is False
    assert result["independent_oos"] is False
    assert result["prereg_commit"] == "690377e"
    with patch.object(runner, "_ancestor", return_value=True):
        with pytest.raises(FileExistsError, match="already sealed"):
            runner.run()


def test_crypto_data_failure_retains_fed_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=True
    ), patch.object(runner, "_current_commit", return_value="synthetic-code"), patch.object(
        runner, "load_fed", return_value=_fed()
    ), patch.object(
        runner, "load_crypto", side_effect=runner.ArchiveIntegrityError("bad bar")
    ):
        path = runner.run()
    result = json.loads(path.read_text())
    assert result["sources"]["fed_h41"]["release_count"] == 269
    assert result["pnl_computed"] is False


def test_margin_failure_and_unexpected_code_failure_both_seal(tmp_path, monkeypatch):
    risk_dir = tmp_path / "risk"
    monkeypatch.setattr(runner, "REPORT_DIR", risk_dir)
    breach = runner.MarginRuinError(pd.Timestamp("2022-01-02", tz="UTC"),
                                    100.0, -20.0, "synthetic ruin")
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=True
    ), patch.object(runner, "_current_commit", return_value="synthetic-code"), patch.object(
        runner, "load_fed", return_value=_fed()
    ), patch.object(runner, "load_crypto", return_value=_crypto()), patch.object(
        runner, "run_backtest", side_effect=breach
    ):
        risk_path = runner.run()
    assert json.loads(risk_path.read_text())["status"] == "sealed-risk-failure"

    code_dir = tmp_path / "code"
    monkeypatch.setattr(runner, "REPORT_DIR", code_dir)
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=True
    ), patch.object(runner, "_current_commit", return_value="synthetic-code"), patch.object(
        runner, "load_fed", side_effect=RuntimeError("synthetic parser bug")
    ):
        code_path = runner.run()
    assert code_path.name == "CODE_FAILURE.json"
    assert json.loads(code_path.read_text())["all_gate_passed"] is False


def test_completed_path_runs_registered_scenarios_once(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    primary, stress, benchmark = _ledger(12_000.0), _ledger(11_000.0), _ledger(9_000.0)
    fed, crypto = _fed(), _crypto()
    with patch.object(runner, "_ancestor", return_value=True), patch.object(
        runner, "_committed_and_pushed", return_value=True
    ), patch.object(runner, "_current_commit", return_value="synthetic-code"), patch.object(
        runner, "load_fed", return_value=fed
    ), patch.object(runner, "load_crypto", return_value=crypto), patch.object(
        runner, "run_backtest", side_effect=[primary, stress, benchmark]
    ) as simulate, patch.object(runner, "evaluate", return_value={
        "independent_oos": False, "full_gate": False,
    }) as metrics:
        path = runner.run()
    assert simulate.call_count == 3
    assert [call.kwargs for call in simulate.call_args_list] == [
        {"slippage_rate": 0.001}, {"slippage_rate": 0.0025},
        {"slippage_rate": 0.001, "benchmark": True},
    ]
    metrics.assert_called_once_with(primary, benchmark, crypto.trade_bars["BTCUSDT"])
    result = json.loads(path.read_text())
    assert result["status"] == "sealed-result"
    assert result["sources"]["fed_h41"]["release_count"] == 269
    assert result["sources"]["binance_usdm"]["archive_count"] == 951
    assert result["account"]["final_usdt"] == 12_000.0
