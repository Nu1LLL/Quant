"""One-shot formal runner for preregistered ETH DeFi TVL study. No orders."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.eth_defi_tvl_source import SourceFailure, load_sources
from research.eth_defi_tvl_sim import evaluate, simulate


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Spot_Multi_Strategy/reports/mini_medallion_eth_defi_tvl_flow_oos"
PREREGS = ("ad00eed", "110964e")
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("this experiment is already sealed")
    for commit in PREREGS:
        result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                cwd=ROOT, check=False, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"missing preregistration ancestor: {commit}")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("implementation must be committed before formal data read")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "refs/remotes/origin/main") != head:
        raise RuntimeError("implementation must be pushed before formal data read")
    return head


def _write_once(name: str, payload: dict) -> Path:
    body = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    with path.open("x", encoding="utf-8") as handle:
        handle.write(body)
    return path


def run() -> Path:
    head = _preflight()
    started = datetime.now(timezone.utc).isoformat()
    base = {"experiment": "eth_defi_tvl_relative_flow_2026_11_08",
            "started_utc": started, "code_commit": head,
            "prereg_commits": list(PREREGS),
            "evidence_grade": "preregistered_retrospective_backfilled_historical_holdout",
            "independent_executable_oos": False, "real_orders_placed": False}
    try:
        tvl, prices, terminal_open, audit = load_sources()
        baseline = simulate(tvl, prices, terminal_open)
        gross = simulate(tvl, prices, terminal_open, slippage=0, fee_rate=0)
        stress_slip = simulate(tvl, prices, terminal_open, slippage=.001)
        stress_latency = simulate(tvl, prices, terminal_open, latency_days=1)
        result = evaluate(baseline, gross, stress_slip, stress_latency, prices)
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
