"""One-shot formal run of the preregistered BTC mempool-demand study."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.btc_exchange_supply_source import SourceFailure
from research.btc_mempool_congestion_source import load_sources
from research.btc_mempool_congestion_sim import evaluate, simulate


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Spot_Multi_Strategy/reports/mini_medallion_btc_mempool_congestion"
PREREG = "fb8f486"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("BTC mempool-demand hypothesis already sealed")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                      cwd=ROOT, check=False, capture_output=True).returncode:
        raise RuntimeError("mempool preregistration absent from implementation ancestry")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit implementation before reading formal data")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "refs/remotes/origin/main") != head:
        raise RuntimeError("push implementation before reading formal data")
    return head


def _write_once(name: str, value: dict) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def run() -> Path:
    code_commit = _preflight()
    base = {"experiment": "btc_mempool_congestion_demand_2026_11_08",
            "prereg_commit": PREREG, "code_commit": code_commit,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_grade": "preregistered_retrospective_current_vintage_blockchain_com_mempool",
            "independent_executable_oos": False, "real_orders_placed": False}
    try:
        medians, counts, prices, terminal, audit = load_sources()
        primary = simulate(medians, counts, prices, terminal)
        gross = simulate(medians, counts, prices, terminal, slippage=0, fee_rate=0)
        double_slip = simulate(medians, counts, prices, terminal, slippage=.001)
        delay = simulate(medians, counts, prices, terminal, latency_days=1)
        hold = simulate(medians, counts, prices, terminal, buy_and_hold=True)
        result = evaluate(primary, gross, double_slip, delay, hold, prices)
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
