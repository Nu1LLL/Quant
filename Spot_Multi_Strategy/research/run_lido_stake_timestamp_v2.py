"""One formal pinned run of the preregistered Lido timestamp-corrected version."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.eth_defi_tvl_source import SourceFailure
from research.lido_stake_source_v2 import load_spot, load_stake
from research.lido_stake_sim import FEE, SLIP, ReplayFailure, evaluate, simulate
from research.stablecoin_liquidity_source import LiquiditySourceFailure


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Spot_Multi_Strategy/reports/mini_medallion_lido_stake_timestamp_v2_oos"
PREREG = "2d11bf0"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("timestamp-corrected Lido version already sealed")
    check = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                           cwd=ROOT, check=False, capture_output=True)
    if check.returncode:
        raise RuntimeError("preregistration not in implementation ancestry")
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
    base = {"experiment": "lido_stake_acceleration_eth_timestamp_v2_2026_11_08",
            "prereg_commit": PREREG, "code_commit": code_commit,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_grade": "preregistered_retrospective_current_lido_vintage",
            "independent_executable_oos": False, "real_orders_placed": False,
            "previous_data_failure_seal": "15ec3b8"}
    try:
        stake, stake_audit = load_stake()
        spot, terminal, spot_audit = load_spot()
        net = simulate(stake, spot, terminal)
        gross = simulate(stake, spot, terminal, slip=0.0, fee_rate=0.0)
        double_slip = simulate(stake, spot, terminal, slip=2 * SLIP)
        latency = simulate(stake, spot, terminal, latency_days=1)
        benchmark = simulate(stake, spot, terminal, slip=SLIP, fee_rate=FEE,
                             benchmark=True)
        result = evaluate(net, gross, double_slip, latency, benchmark, spot)
        if len(spot_audit) != 63:
            raise SourceFailure("ETH spot audit count incomplete")
        return _write_once("RESULT.json", {
            **base, "status": "sealed_formal_result",
            "sealed_utc": datetime.now(timezone.utc).isoformat(),
            "sources": {"lido": stake_audit, "binance_spot_verified_sha256": spot_audit},
            **result})
    except (SourceFailure, ReplayFailure, LiquiditySourceFailure) as exc:
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
