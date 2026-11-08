"""One-shot sealed run of preregistered CME BTC option-pressure residual."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.cme_option_pressure_source import SourceFailure, load_sources
from research.cme_option_pressure_sim import evaluate, simulate


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Spot_Multi_Strategy/reports/mini_medallion_cme_option_pressure_oos"
PREREG = "f900f63"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("CME BTC option-pressure hypothesis already sealed")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                              cwd=ROOT, check=False, capture_output=True)
    if ancestor.returncode:
        raise RuntimeError("preregistration is not an ancestor of implementation")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit all implementation before opening formal data")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "refs/remotes/origin/main") != head:
        raise RuntimeError("push implementation before opening formal data")
    return head


def _write_once(name: str, value: dict) -> Path:
    body = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    with path.open("x", encoding="utf-8") as handle:
        handle.write(body)
    return path


def run() -> Path:
    code_commit = _preflight()
    base = {"experiment": "cme_btc_option_nonmomentum_pressure_2026_11_08",
            "prereg_commit": PREREG, "code_commit": code_commit,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_grade": "preregistered_current_vintage_retrospective_option_pressure",
            "independent_executable_oos": False, "real_orders_placed": False}
    try:
        reports, prices, terminal_open, audit = load_sources()
        primary = simulate(reports, prices, terminal_open)
        gross = simulate(reports, prices, terminal_open, slippage=0, fee_rate=0)
        double_slip = simulate(reports, prices, terminal_open, slippage=.001)
        extra_day = simulate(reports, prices, terminal_open, latency_days=1)
        hold = simulate(reports, prices, terminal_open, buy_and_hold=True)
        result = evaluate(primary, gross, double_slip, extra_day, hold, prices)
        return _write_once("RESULT.json", {**base, "status": "sealed_formal_result",
                                           "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                           "sources": audit, **result})
    except SourceFailure as exc:
        return _write_once("DATA_FAILURE.json", {**base, "status": "sealed_source_or_execution_failure",
                                                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                                 "reason": str(exc), "error_type": type(exc).__name__,
                                                 "gate_passed": False})
    except Exception as exc:
        return _write_once("CODE_FAILURE.json", {**base, "status": "sealed_code_failure",
                                                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                                 "reason": str(exc), "error_type": type(exc).__name__,
                                                 "gate_passed": False})


if __name__ == "__main__":
    print(run())
