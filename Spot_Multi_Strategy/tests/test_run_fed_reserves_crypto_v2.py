"""Fixture-only v2 orchestration checks; no actual H.4.1 or Binance fetches."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from research import run_fed_reserves_crypto as v1
from research import run_fed_reserves_crypto_v2 as runner


def _fed():
    return SimpleNamespace(observations=(object(),), manifest={"release_count": 269})


def _crypto():
    return SimpleNamespace(
        trade_bars={"BTCUSDT": object()}, mark_bars={"BTCUSDT": object()},
        funding_events={"BTCUSDT": object()}, manifest={"archive_count": 951},
    )


def _ledger(final):
    return SimpleNamespace(
        final_equity=final, fee_usdt=10.0, slippage_usdt=10.0,
        funding_usdt=1.0, turnover_usdt=10_000.0,
        conservative_max_drawdown=-0.10,
        trades=(object(),), records=(SimpleNamespace(exposed=True),),
    )


def test_only_source_parser_changes_from_v1():
    assert runner.load_crypto is v1.load_crypto
    assert runner.run_backtest is v1.run_backtest
    assert runner.evaluate is v1.evaluate
    assert runner.load_fed is not v1.load_fed
    assert runner.PREREG_COMMIT == "478b552"
    assert runner.REPORT_DIR != v1.REPORT_DIR


def test_prereg_clean_push_and_one_shot_guard_precede_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    with patch.object(v1, "_ancestor", return_value=False), patch.object(
        runner, "load_fed"
    ) as fetch:
        with pytest.raises(RuntimeError, match="preregistration"):
            runner.run()
        fetch.assert_not_called()
    with patch.object(v1, "_ancestor", return_value=True), patch.object(
        v1, "_committed_and_pushed", return_value=False
    ), patch.object(runner, "load_fed") as fetch:
        with pytest.raises(RuntimeError, match="committed, pushed"):
            runner.run()
        fetch.assert_not_called()
    for seal in ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json"):
        path = tmp_path / seal
        path.write_text("sealed")
        with patch.object(v1, "_ancestor", return_value=True), patch.object(
            runner, "load_fed"
        ) as fetch:
            with pytest.raises(FileExistsError, match="already sealed"):
                runner.run()
            fetch.assert_not_called()
        path.unlink()


def test_fed_source_failure_seals_v2_without_crypto_fetch_or_pnl(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    with patch.object(v1, "_ancestor", return_value=True), patch.object(
        v1, "_committed_and_pushed", return_value=True
    ), patch.object(v1, "_current_commit", return_value="synthetic-code"), patch.object(
        runner, "load_fed", side_effect=runner.VintageDataError("bad fixture")
    ), patch.object(runner, "load_crypto") as crypto, patch.object(
        runner, "run_backtest"
    ) as simulate:
        path = runner.run()
        crypto.assert_not_called()
        simulate.assert_not_called()
    result = json.loads(path.read_text())
    assert path.name == "DATA_FAILURE.json"
    assert result["prereg_commit"] == "478b552"
    assert result["previous_sealed_attempt"] == "3954675"
    assert result["pnl_computed"] is False
    assert result["independent_oos"] is False
    assert result["evidence_grade"] == "preregistered_retrospective_historical_holdout"


def test_complete_scenarios_preserve_v1_parameters_and_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    primary, stress, benchmark = _ledger(12_000), _ledger(11_000), _ledger(9_000)
    crypto = _crypto()
    with patch.object(v1, "_ancestor", return_value=True), patch.object(
        v1, "_committed_and_pushed", return_value=True
    ), patch.object(v1, "_current_commit", return_value="synthetic-code"), patch.object(
        runner, "load_fed", return_value=_fed()
    ), patch.object(runner, "load_crypto", return_value=crypto), patch.object(
        runner, "run_backtest", side_effect=[primary, stress, benchmark]
    ) as simulate, patch.object(runner, "evaluate", return_value={
        "independent_oos": False, "full_gate": False,
    }) as metrics:
        path = runner.run()
    assert [call.kwargs for call in simulate.call_args_list] == [
        {"slippage_rate": 0.001}, {"slippage_rate": 0.0025},
        {"slippage_rate": 0.001, "benchmark": True},
    ]
    metrics.assert_called_once_with(primary, benchmark, crypto.trade_bars["BTCUSDT"])
    result = json.loads(path.read_text())
    assert result["status"] == "sealed-result"
    assert result["sources"]["fed_h41"]["release_count"] == 269
    assert result["account"]["final_usdt"] == 12_000
    assert result["same_economic_hypothesis_as_v1"] is True
    assert result["all_gate_passed"] is False
    assert result["independent_oos"] is False
    assert result["evidence_grade"] == "preregistered_retrospective_historical_holdout"


def test_margin_and_unexpected_code_failures_are_sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path / "margin")
    breach = runner.MarginRuinError(pd.Timestamp("2022-01-02", tz="UTC"),
                                    100.0, -20.0, "synthetic ruin")
    with patch.object(v1, "_ancestor", return_value=True), patch.object(
        v1, "_committed_and_pushed", return_value=True
    ), patch.object(v1, "_current_commit", return_value="synthetic-code"), patch.object(
        runner, "load_fed", return_value=_fed()
    ), patch.object(runner, "load_crypto", return_value=_crypto()), patch.object(
        runner, "run_backtest", side_effect=breach
    ):
        path = runner.run()
    assert json.loads(path.read_text())["status"] == "sealed-risk-failure"

    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path / "code")
    with patch.object(v1, "_ancestor", return_value=True), patch.object(
        v1, "_committed_and_pushed", return_value=True
    ), patch.object(v1, "_current_commit", return_value="synthetic-code"), patch.object(
        runner, "load_fed", side_effect=RuntimeError("synthetic parser bug")
    ):
        path = runner.run()
    assert path.name == "CODE_FAILURE.json"
    assert json.loads(path.read_text())["all_gate_passed"] is False
