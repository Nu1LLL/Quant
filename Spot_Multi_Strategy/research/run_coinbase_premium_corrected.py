"""One-shot official-history run for the preregistered Coinbase premium rule."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.coinbase_premium_source import SourceFailure, load_sources
from research.coinbase_premium_sim import evaluate, premium_series, simulate


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Spot_Multi_Strategy/reports/mini_medallion_coinbase_premium_corrected_oos"
PREREG = "e63231d"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("this historical hypothesis is already sealed")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                              cwd=ROOT, check=False, capture_output=True)
    if ancestor.returncode:
        raise RuntimeError("the preregistration is not an ancestor of code")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit implementation before opening formal prices")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "refs/remotes/origin/main") != head:
        raise RuntimeError("push implementation before opening formal prices")
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
    base = {"experiment": "coinbase_premium_corrected_btc_2026_11_08",
            "prereg_commit": PREREG, "code_commit": code_commit,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_grade": "preregistered_retrospective_cross_venue_temporal_holdout",
            "independent_executable_oos": False, "real_orders_placed": False}
    try:
        cb_btc, cb_usdt, binance, terminal_open, sources = load_sources()
        premium = premium_series(cb_btc, cb_usdt, binance)
        baseline = simulate(premium, binance, terminal_open)
        gross = simulate(premium, binance, terminal_open, slippage=0, fee_rate=0)
        stress_slip = simulate(premium, binance, terminal_open, slippage=.001)
        stress_latency = simulate(premium, binance, terminal_open, latency_days=1)
        result = evaluate(baseline, gross, stress_slip, stress_latency, binance)
        return _write_once("RESULT.json", {**base, "status": "sealed_formal_result",
                                           "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                           "sources": sources, **result})
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
