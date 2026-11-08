"""Run and seal the single preregistered Deribit DVOL basket experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.dvol_risk_source import DvolSourceFailure, load_dvol, load_spot
from research.dvol_risk_sim import (
    DvolReplayFailure, FEE, SLIP, evaluate, simulate,
)
from research.stablecoin_liquidity_source import LiquiditySourceFailure


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Spot_Multi_Strategy/reports/mini_medallion_deribit_dvol_risk_premium_oos"
PREREG = "5e1f4bf"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("DVOL risk-premium hypothesis already sealed")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                              cwd=ROOT, check=False, capture_output=True)
    if ancestor.returncode:
        raise RuntimeError("preregistration is not an implementation ancestor")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit implementation before reading economic inputs")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "refs/remotes/origin/main") != head:
        raise RuntimeError("push implementation before reading economic inputs")
    return head


def _write_once(name: str, payload: dict) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return path


def run() -> Path:
    code_commit = _preflight()
    base = {"experiment": "deribit_dvol_risk_premium_basket_2026_11_08",
            "prereg_commit": PREREG, "code_commit": code_commit,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_grade": "preregistered_retrospective_current_dvol_vintage",
            "independent_executable_oos": False, "real_orders_placed": False}
    try:
        dvol, dvol_audit = load_dvol()
        spot, terminal, spot_audit = load_spot()
        net = simulate(dvol, spot, terminal)
        gross = simulate(dvol, spot, terminal, slip=0.0, fee_rate=0.0)
        double_slip = simulate(dvol, spot, terminal, slip=2 * SLIP)
        latency = simulate(dvol, spot, terminal, latency_days=1)
        benchmark = simulate(dvol, spot, terminal, fee_rate=FEE, slip=SLIP,
                             benchmark=True)
        result = evaluate(net, gross, double_slip, latency, benchmark, spot)
        if len(dvol_audit) != 12 or len(spot_audit) != 136:
            raise DvolSourceFailure("source audit record count incomplete")
        return _write_once("RESULT.json", {**base, "status": "sealed_formal_result",
                                           "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                           "sources": {"deribit_dvol_verified_sha256": dvol_audit,
                                                       "binance_spot_verified_sha256": spot_audit},
                                           **result})
    except (DvolSourceFailure, DvolReplayFailure, LiquiditySourceFailure) as exc:
        return _write_once("DATA_FAILURE.json", {
            **base, "status": "sealed_source_or_execution_failure",
            "sealed_utc": datetime.now(timezone.utc).isoformat(),
            "reason": str(exc), "error_type": type(exc).__name__, "gate_passed": False})
    except Exception as exc:
        return _write_once("CODE_FAILURE.json", {
            **base, "status": "sealed_code_failure",
            "sealed_utc": datetime.now(timezone.utc).isoformat(),
            "reason": str(exc), "error_type": type(exc).__name__, "gate_passed": False})


if __name__ == "__main__":
    print(run())
