"""Single formal sealed run of the preregistered stablecoin-supply regime."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.stablecoin_liquidity_source import (
    LiquiditySourceFailure, load_spot, load_stablecoins,
)
from research.stablecoin_liquidity_sim import (
    FEE, SLIP, ReplayFailure, evaluate, simulate,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (ROOT / "Spot_Multi_Strategy/reports/"
          "mini_medallion_usdt_usdc_liquidity_regime_oos")
PREREG = "51185fe"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("stablecoin liquidity historical hypothesis already sealed")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                              cwd=ROOT, check=False, capture_output=True)
    if ancestor.returncode:
        raise RuntimeError("preregistration is not an implementation ancestor")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit the implementation before reading economic inputs")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "refs/remotes/origin/main") != head:
        raise RuntimeError("push implementation before reading economic inputs")
    return head


def _write_once(name: str, value: dict) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return path


def run() -> Path:
    code_commit = _preflight()
    base = {"experiment": "usdt_usdc_liquidity_regime_2026_11_08",
            "prereg_commit": PREREG, "code_commit": code_commit,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_grade": "preregistered_retrospective_current_stablecoin_vintage",
            "independent_executable_oos": False, "real_orders_placed": False}
    try:
        stablecoins, stable_audit = load_stablecoins()
        spot, terminal, spot_audit = load_spot()
        net = simulate(stablecoins, spot, terminal)
        gross = simulate(stablecoins, spot, terminal, slip=0.0, fee_rate=0.0)
        stress_slip = simulate(stablecoins, spot, terminal, slip=2 * SLIP)
        stress_latency = simulate(stablecoins, spot, terminal, latency_days=1)
        benchmark = simulate(stablecoins, spot, terminal, benchmark=True,
                             slip=SLIP, fee_rate=FEE)
        result = evaluate(net, gross, stress_slip, stress_latency, benchmark, spot)
        if len(spot_audit) != 186 or len(stable_audit) != 2:
            raise LiquiditySourceFailure("source audit record count incomplete")
        return _write_once("RESULT.json", {**base, "status": "sealed_formal_result",
                                           "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                           "sources": {"stablecoins": stable_audit,
                                                       "binance_spot_verified_sha256": spot_audit},
                                           **result})
    except (LiquiditySourceFailure, ReplayFailure) as exc:
        return _write_once("DATA_FAILURE.json", {**base,
                                                 "status": "sealed_source_or_execution_failure",
                                                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                                 "reason": str(exc),
                                                 "error_type": type(exc).__name__,
                                                 "gate_passed": False})
    except Exception as exc:
        return _write_once("CODE_FAILURE.json", {**base,
                                                 "status": "sealed_code_failure",
                                                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                                 "reason": str(exc),
                                                 "error_type": type(exc).__name__,
                                                 "gate_passed": False})


if __name__ == "__main__":
    print(run())
