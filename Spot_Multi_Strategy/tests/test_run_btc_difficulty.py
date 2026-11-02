import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from Spot_Multi_Strategy.research import run_btc_difficulty as runner
from Spot_Multi_Strategy.research.btc_difficulty_blocks import DifficultySourceError


@dataclass(frozen=True)
class BlockManifest:
    url: str


@dataclass(frozen=True)
class SpotManifest:
    month: str


def test_source_failure_seals_without_pnl(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(runner, "_ancestor", lambda commit: True)
    monkeypatch.setattr(runner, "_commit", lambda: "test-code")

    def block_failure():
        raise DifficultySourceError("no verified block page")

    monkeypatch.setattr(runner, "fetch_events", block_failure)
    monkeypatch.setattr(runner, "fetch_bars", lambda: object())
    monkeypatch.setattr(runner, "simulate", lambda *args, **kwargs: pytest.fail("PnL ran"))
    path = runner.run()
    assert path.name == "DATA_FAILURE.json"
    result = json.loads(path.read_text())
    assert result["pnl_computed"] is False
    assert result["independent_oos"] is False
    assert "account" not in result
    with pytest.raises(FileExistsError):
        runner.run()


def test_success_writes_compact_result_once(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(runner, "_ancestor", lambda commit: True)
    monkeypatch.setattr(runner, "_commit", lambda: "test-code")
    event = SimpleNamespace(recovery=True)
    block_manifest = BlockManifest("https://blockstream.info/api/blocks/1")
    spot_manifest = SpotManifest("2026-09")
    monkeypatch.setattr(runner, "fetch_events", lambda: SimpleNamespace(
        events=(event,), manifest=(block_manifest,)))
    monkeypatch.setattr(runner, "fetch_bars", lambda: SimpleNamespace(
        bars=(object(),), manifest=(spot_manifest,)))
    record = SimpleNamespace(exposed=True)
    ledger = SimpleNamespace(
        eligible_recovery_events=1, ignored_events_while_held=0,
        trades=(object(), object()), records=(record,), final_equity=10_100,
        conservative_max_drawdown=-0.03, capacity_pass=True,
    )
    monkeypatch.setattr(runner, "simulate", lambda *args, **kwargs: ledger)
    monkeypatch.setattr(runner, "buy_and_hold", lambda *args, **kwargs: ledger)
    monkeypatch.setattr(runner, "evaluate_diagnostics", lambda *args: {
        "numerical_gate": False, "full_gate": False,
    })
    path = runner.run()
    assert path.name == "RESULT.json"
    result = json.loads(path.read_text())
    assert result["status"] == "sealed-result"
    assert result["independent_oos"] is False
    assert result["counts"]["trades"] == 2
    assert not (tmp_path / "DATA_FAILURE.json").exists()
    with pytest.raises(FileExistsError):
        runner.run()
