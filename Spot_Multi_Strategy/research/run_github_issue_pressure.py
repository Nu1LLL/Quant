"""Run and seal one preregistered GitHub-issue BTC/ETH experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .github_issue_pressure_data import load_pair_tape
from .github_issue_pressure_sim import (
    StrategyFailure, evaluate, load_fixed_counts, schedule,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "Spot_Multi_Strategy" / "research" / "results"
INPUT = RESULT_DIR / "github_issue_pressure_source_2026_11_08.json"
OUTPUT = RESULT_DIR / "github_issue_pressure_result_2026_11_08.json"
PREREG = "8cb7cdb"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run() -> Path:
    if subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                      cwd=ROOT, check=False).returncode:
        raise RuntimeError("preregistration must precede the run")
    if OUTPUT.exists():
        raise FileExistsError("this experiment is already sealed")
    started = datetime.now(timezone.utc).isoformat()
    counts, count_hash = load_fixed_counts(INPUT)
    decisions = schedule(counts)
    base = {
        "id": "github_issue_pressure_btc_eth_2026_11_08",
        "started_utc": started,
        "sealed_utc": None,
        "prereg_commit": PREREG,
        "code_commit": _git("rev-parse", "HEAD"),
        "source_count_sha256": count_hash,
        "evidence_grade": "preregistered_retrospective_current_github_search_index",
        "independent_executable_oos": False,
    }
    try:
        tape = load_pair_tape()
    except ValueError as exc:
        payload = {**base, "status": "source_integrity_fail", "error": str(exc),
                   "error_type": type(exc).__name__, "pnl_computed": False,
                   "all_gate_passed": False}
    else:
        sources = {"price_archives": len(tape.price_hashes),
                   "funding_archives": len(tape.funding_hashes),
                   "price_sha256": tape.price_hashes,
                   "funding_sha256": tape.funding_hashes,
                   "five_minute_bars": len(tape.opening),
                   "funding_event_bars": len(tape.funding)}
        try:
            result = evaluate(tape, decisions)
        except StrategyFailure as exc:
            payload = {**base, "status": "sealed_strategy_failure",
                       "error": str(exc), "error_type": type(exc).__name__,
                       "bar": exc.bar, "equity_at_failure": exc.equity,
                       "sources": sources, "all_gate_passed": False}
        else:
            payload = {**base, "status": "sealed_result", "sources": sources,
                       "result": result,
                       "all_gate_passed": bool(result["gate"]["all_passed"])}
    payload["sealed_utc"] = datetime.now(timezone.utc).isoformat()
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return OUTPUT


if __name__ == "__main__":
    print(run())
