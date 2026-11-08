"""One-shot formal historical run of the frozen Deribit BTC IV term slope."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.btc_exchange_supply_source import SourceFailure
from research.deribit_iv_term_source import load_sources
from research.deribit_iv_term_sim import evaluate, simulate


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Spot_Multi_Strategy/reports/mini_medallion_deribit_iv_term_oos"
PREREG = "7a31ecc"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("Deribit BTC IV term-structure hypothesis already sealed")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                      cwd=ROOT, check=False, capture_output=True).returncode:
        raise RuntimeError("preregistration absent from implementation history")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit implementation before reading formal source")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "refs/remotes/origin/main") != head:
        raise RuntimeError("push implementation before reading formal source")
    return head


def _write_once(name: str, value: dict) -> Path:
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    with path.open("x", encoding="utf-8") as handle:
        handle.write(body)
    return path


def run() -> Path:
    code_commit = _preflight()
    base = {"experiment": "deribit_btc_iv_term_structure_2026_11_09",
            "prereg_commit": PREREG, "code_commit": code_commit,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_grade": "preregistered_retrospective_crypto_option_term_structure",
            "independent_executable_oos": False, "real_orders_placed": False}
    try:
        targets, prices, terminal, audit = load_sources()
        primary = simulate(targets, prices, terminal)
        gross = simulate(targets, prices, terminal, slippage=0, fee_rate=0)
        double = simulate(targets, prices, terminal, slippage=.001)
        late = simulate(targets, prices, terminal, latency_days=1)
        hold = simulate(targets, prices, terminal, buy_and_hold=True)
        result = evaluate(primary, gross, double, late, hold, prices, audit)
        return _write_once("RESULT.json", {**base, "status": "sealed_formal_result",
                                           "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                           "sources": audit, **result})
    except SourceFailure as exc:
        return _write_once("DATA_FAILURE.json", {**base,
                                                 "status": "sealed_source_or_execution_failure",
                                                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                                 "reason": str(exc), "error_type": type(exc).__name__,
                                                 "gate_passed": False})
    except Exception as exc:
        return _write_once("CODE_FAILURE.json", {**base,
                                                 "status": "sealed_code_failure",
                                                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                                 "reason": str(exc), "error_type": type(exc).__name__,
                                                 "gate_passed": False})


if __name__ == "__main__":
    print(run())
