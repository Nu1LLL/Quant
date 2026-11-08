"""One-shot formal result seal for three DeFi parent-revenue surprises."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.basis_hourly_preflight import PreflightError
from research.defi_revenue_spread_data import (
    DefiRevenueDataError, iter_hours, load_funding,
)
from research.defi_revenue_spread_sim import ReplayFailure, evaluate, run_replay
from research.defi_revenue_spread_source import RevenueSourceFailure, load_revenue


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Spot_Multi_Strategy/reports/mini_medallion_three_defi_revenue_surprise_spread_oos"
PREREG = "433422d"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("DeFi revenue spread historical hypothesis already sealed")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                              cwd=ROOT, check=False, capture_output=True)
    if ancestor.returncode:
        raise RuntimeError("preregistration is not an ancestor of implementation")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit implementation before opening formal data")
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
    base = {"experiment": "three_defi_revenue_surprise_spread_2026_11_08",
            "prereg_commit": PREREG, "code_commit": code_commit,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_grade": "preregistered_retrospective_current_protocol_vintage",
            "independent_executable_oos": False, "real_orders_placed": False}
    try:
        revenue, revenue_audit = load_revenue()
        funding_audit: dict[tuple[str, str], str] = {}
        events = load_funding(audit_hashes=funding_audit)
        price_audit: dict[tuple[str, str], str] = {}
        accounts, states, execution = run_replay(
            revenue, iter_hours(audit_hashes=price_audit), events)
        result = evaluate(accounts, states, execution)
        sources = {"defillama": revenue_audit,
                   "binance_hourly_verified_sha256": [
                       {"month": key[0], "symbol": key[1], "sha256": value}
                       for key, value in sorted(price_audit.items())],
                   "binance_funding_verified_sha256": [
                       {"month": key[0], "symbol": key[1], "sha256": value}
                       for key, value in sorted(funding_audit.items())]}
        if len(price_audit) != 186 or len(funding_audit) != 183:
            raise DefiRevenueDataError("source archive audit count incomplete")
        return _write_once("RESULT.json", {**base, "status": "sealed_formal_result",
                                           "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                           "sources": sources, **result})
    except (RevenueSourceFailure, DefiRevenueDataError, PreflightError, ReplayFailure) as exc:
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
